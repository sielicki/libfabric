/*
 * Example: NCCL-like Plugin Using CXI KI (Kernel Initiated) API
 *
 * This example demonstrates how a GPU communication library (like NCCL)
 * would use the KI API to enable GPU kernels to directly post RDMA
 * operations without CPU involvement.
 *
 * Architecture:
 *   1. Host setup phase: Use libfabric to create endpoints, register memory,
 *      resolve peers, and populate KI metadata
 *   2. GPU kernel phase: Kernels directly post PUT/GET/atomic operations
 *      using the KI device API
 *   3. Completion: GPU polls writebacks or signals for completion
 */

#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_cxi_ext.h>

/* Include the KI GPU header for device code */
#include "fi_cxi_ki_gpu.h"

/*
 * =============================================================================
 * Host-Side Setup Code
 * =============================================================================
 */

struct nccl_ki_comm {
	/* Libfabric handles */
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_av *av;
	struct fid_mr *send_mr;
	struct fid_mr *recv_mr;
	struct fid_cntr **counters;

	/* KI operations */
	struct fi_cxi_ki_ops *ki_ops;

	/* GPU-accessible metadata (allocated with cudaMallocManaged) */
	struct fi_cxi_ki_meta *ki_meta;

	/* Peer info */
	int num_peers;
	fi_addr_t *peer_addrs;
};

/*
 * Initialize KI metadata for GPU access.
 * This is called once during communicator setup.
 */
int nccl_ki_init(struct nccl_ki_comm *comm, int num_peers, int num_counters)
{
	int ret;

	/* Get KI operations from the endpoint */
	ret = fi_open_ops(&comm->ep->fid, FI_CXI_KI_OPS_1, 0,
			  (void **)&comm->ki_ops, NULL);
	if (ret)
		return ret;

	/* Allocate GPU-accessible metadata using managed memory */
	cudaMallocManaged(&comm->ki_meta, sizeof(struct fi_cxi_ki_meta));
	memset(comm->ki_meta, 0, sizeof(struct fi_cxi_ki_meta));

	/* Get endpoint info */
	struct fi_cxi_ki_ep_info ep_info;
	ret = comm->ki_ops->get_ep_info(&comm->ep->fid, &ep_info);
	if (ret)
		return ret;

	comm->ki_meta->config.nid = ep_info.nid;
	comm->ki_meta->config.pid = ep_info.pid;
	comm->ki_meta->config.pid_bits = ep_info.pid_bits;
	comm->ki_meta->config.vni = ep_info.vni;

	/* Get command queue info */
	struct fi_cxi_ki_cmdq_info cmdq_info;
	ret = comm->ki_ops->get_cmdq_info(&comm->ep->fid, 0, &cmdq_info);
	if (ret)
		return ret;

	/* Allocate cmdq config/hot arrays (1 context for simplicity) */
	cudaMallocManaged(&comm->ki_meta->cmdq_config,
			  sizeof(struct fi_cxi_ki_cmdq_config));
	cudaMallocManaged(&comm->ki_meta->cmdq_hot,
			  sizeof(struct fi_cxi_ki_cmdq_hot));

	comm->ki_meta->cmdq_config[0].buf = cmdq_info.cmdq_buf;
	comm->ki_meta->cmdq_config[0].size = cmdq_info.cmdq_size;
	comm->ki_meta->cmdq_config[0].mask = cmdq_info.cmdq_mask;
	comm->ki_meta->cmdq_config[0].slot_size = cmdq_info.cmd_slot_size;
	comm->ki_meta->cmdq_hot[0].wp_addr = cmdq_info.wp_addr;
	comm->ki_meta->cmdq_hot[0].wp = 0;
	comm->ki_meta->config.context_count = 1;

	/* Get local MR info for LAC */
	struct fi_cxi_ki_mr_info mr_info;
	ret = comm->ki_ops->get_mr_info(&comm->send_mr->fid, &mr_info);
	if (ret)
		return ret;
	comm->ki_meta->config.local_lac = mr_info.lac;

	/* Allocate peer arrays (SoA layout for coalesced GPU access) */
	comm->num_peers = num_peers;
	cudaMallocManaged(&comm->ki_meta->peers.dfa,
			  num_peers * sizeof(uint32_t));
	cudaMallocManaged(&comm->ki_meta->peers.dfa_ext,
			  num_peers * sizeof(uint8_t));
	cudaMallocManaged(&comm->ki_meta->peers.index_ext,
			  num_peers * sizeof(uint8_t));
	cudaMallocManaged(&comm->ki_meta->peers.mr_base,
			  num_peers * sizeof(uint64_t));
	cudaMallocManaged(&comm->ki_meta->peers.mr_key,
			  num_peers * sizeof(uint64_t));
	cudaMallocManaged(&comm->ki_meta->peers.signal_base,
			  num_peers * sizeof(uint64_t));
	cudaMallocManaged(&comm->ki_meta->peers.signal_key,
			  num_peers * sizeof(uint64_t));
	comm->ki_meta->peers.count = num_peers;

	/* Resolve each peer */
	for (int i = 0; i < num_peers; i++) {
		struct fi_cxi_ki_target_info target;
		ret = comm->ki_ops->resolve_target(&comm->ep->fid,
						   comm->peer_addrs[i],
						   &target);
		if (ret)
			return ret;

		comm->ki_meta->peers.dfa[i] = target.dfa;
		comm->ki_meta->peers.dfa_ext[i] = target.dfa_ext;
		comm->ki_meta->peers.index_ext[i] = target.index_ext;

		/* TODO: Exchange remote MR info out-of-band */
		/* comm->ki_meta->peers.mr_base[i] = remote_mr_base; */
		/* comm->ki_meta->peers.mr_key[i] = remote_mr_key; */
	}

	/* Allocate counters with GPU writeback */
	cudaMallocManaged(&comm->ki_meta->wb.cntr_wb,
			  num_counters * sizeof(struct fi_cxi_ki_ct_writeback));

	struct fi_cxi_ki_cntr_batch_info cntr_info;
	comm->counters = (struct fid_cntr **)calloc(num_counters,
						    sizeof(struct fid_cntr *));

	ret = comm->ki_ops->alloc_counters_batch(
		&comm->ep->fid, num_counters,
		(void *)comm->ki_meta->wb.cntr_wb, comm->counters, &cntr_info);
	if (ret)
		return ret;

	cudaMallocManaged(&comm->ki_meta->wb.cntr_ct_idx,
			  num_counters * sizeof(uint16_t));
	memcpy(comm->ki_meta->wb.cntr_ct_idx, cntr_info.ct_indices,
	       num_counters * sizeof(uint16_t));

	comm->ki_meta->wb.cntr_mmio_base = cntr_info.mmio_base;
	comm->ki_meta->wb.cntr_mmio_stride = cntr_info.mmio_stride;
	comm->ki_meta->wb.counter_count = num_counters;

	/* Allocate signals for peer notification */
	int num_signals = num_peers;
	cudaMallocManaged(&comm->ki_meta->wb.signals,
			  num_signals * sizeof(uint64_t));
	memset((void *)comm->ki_meta->wb.signals, 0,
	       num_signals * sizeof(uint64_t));
	comm->ki_meta->wb.signal_count = num_signals;

	return 0;
}

/*
 * =============================================================================
 * GPU Kernel Code - Using KI API
 * =============================================================================
 */

/*
 * Simple PUT kernel - each thread sends its data to a peer.
 * Uses thread-level allocation (1 atomic per thread).
 */
__global__ void ki_put_kernel_simple(fi_ki_meta_t meta, int peer_idx,
				     const void *src, size_t chunk_size,
				     int num_chunks)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_chunks)
		return;

	const char *my_src = (const char *)src + tid * chunk_size;
	uint64_t dst_off = tid * chunk_size;

	/* Post PUT using thread-level allocation */
	fi_cxi_ki::inject_put_simple(meta, 0, peer_idx, my_src, dst_off,
				     chunk_size, 0 /* counter */);
}

/*
 * Warp-cooperative PUT kernel - reduces atomic contention by 32x.
 * All threads in a warp coordinate to allocate slots with a single atomic.
 */
__global__ void ki_put_kernel_warp_coop(fi_ki_meta_t meta, int peer_idx,
					const void *src, size_t chunk_size,
					int num_chunks)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool active = (tid < num_chunks);

	const char *my_src = nullptr;
	uint64_t dst_off = 0;

	if (active) {
		my_src = (const char *)src + tid * chunk_size;
		dst_off = tid * chunk_size;
	}

	/* Warp-cooperative PUT - only 1 atomic per warp instead of 32 */
	fi_cxi_ki::inject_put_simple<fi_cxi_ki::Coop::Warp>(
		meta, 0, peer_idx, my_src, dst_off,
		active ? chunk_size : 0, /* len=0 means don't post */
		active ? 0 : FI_KI_NO_COUNTER);
}

/*
 * Block-cooperative PUT kernel - reduces atomic contention to 1 per block.
 * Uses shared memory for intra-block coordination.
 */
__global__ void ki_put_kernel_block_coop(fi_ki_meta_t meta, int peer_idx,
					 const void *src, size_t chunk_size,
					 int num_chunks)
{
	/* Shared memory for block-level coordination */
	extern __shared__ uint32_t smem[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool active = (tid < num_chunks);

	const char *my_src = nullptr;
	uint64_t dst_off = 0;

	if (active) {
		my_src = (const char *)src + tid * chunk_size;
		dst_off = tid * chunk_size;
	}

	/* Block-cooperative PUT - only 1 atomic per block */
	fi_cxi_ki::inject_put_simple<fi_cxi_ki::Coop::Block>(
		meta, 0, peer_idx, my_src, dst_off,
		active ? chunk_size : 0, active ? 0 : FI_KI_NO_COUNTER, smem);
}

/*
 * Ring allreduce kernel - demonstrates multi-peer communication.
 * Each GPU sends to next peer and receives from previous.
 */
__global__ void ki_ring_allreduce_step(fi_ki_meta_t meta, int send_peer,
				       int recv_peer, const float *send_buf,
				       volatile float *recv_buf,
				       size_t num_elements, int step)
{
	extern __shared__ uint32_t smem[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int num_threads = gridDim.x * blockDim.x;

	/* Each thread handles a chunk of elements */
	size_t elems_per_thread = (num_elements + num_threads - 1) / num_threads;
	size_t my_start = tid * elems_per_thread;
	size_t my_count = min(elems_per_thread, num_elements - my_start);

	if (my_start >= num_elements)
		my_count = 0;

	/* Send our chunk to the next peer */
	const float *my_src = send_buf + my_start;
	uint64_t dst_off = my_start * sizeof(float);
	size_t len = my_count * sizeof(float);

	/* Use block-cooperative allocation for efficiency */
	fi_cxi_ki::inject_put_simple<fi_cxi_ki::Coop::Block>(
		meta, 0, send_peer, my_src, dst_off, len,
		0 /* counter */, smem);

	/* Ring doorbell to initiate transfer */
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		fi_cxi_ring_doorbell(meta, 0);
	}
}

/*
 * Atomic reduction kernel - demonstrates type-safe atomics.
 * Performs remote atomic add to accumulate partial sums.
 */
__global__ void ki_atomic_reduce(fi_ki_meta_t meta, int peer_idx,
				 uint64_t remote_offset,
				 const int64_t *partial_sums, int num_partials)
{
	extern __shared__ uint32_t smem[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= num_partials)
		return;

	int64_t my_value = partial_sums[tid];

	/* Type-safe atomic add - compiler verifies int64_t is supported */
	fi_cxi_ki::atomic_add<int64_t, fi_cxi_ki::Coop::Warp>(
		meta, 0, peer_idx, remote_offset, my_value);
}

/*
 * Signal-based completion kernel - waits for peer signals.
 */
__global__ void ki_wait_signals(fi_ki_meta_t meta, int num_peers,
				uint64_t expected_count)
{
	/* Only one thread needs to poll */
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	/* Wait for all peers to signal completion */
	for (int i = 0; i < num_peers; i++) {
		fi_cxi_signal_wait(meta, i, expected_count);
	}
}

/*
 * Send completion signal to peer after data is ready.
 */
__global__ void ki_send_signal(fi_ki_meta_t meta, int peer_idx, uint32_t sig_idx,
			       uint64_t value)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	fi_cxi_signal_send(meta, 0, peer_idx, sig_idx, value);
	fi_cxi_ring_doorbell(meta, 0);
}

/*
 * =============================================================================
 * Host-Side Launch Code
 * =============================================================================
 */

void nccl_ki_send(struct nccl_ki_comm *comm, int peer, const void *buf,
		  size_t size, cudaStream_t stream)
{
	/* Calculate optimal chunking */
	size_t chunk_size = 4096;
	int num_chunks = (size + chunk_size - 1) / chunk_size;

	/* Use block-cooperative kernel for best performance */
	int threads_per_block = 256;
	int num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;
	size_t smem_size = (threads_per_block / 32 + 2) * sizeof(uint32_t);

	ki_put_kernel_block_coop<<<num_blocks, threads_per_block, smem_size,
				   stream>>>(comm->ki_meta, peer, buf,
					     chunk_size, num_chunks);

	/* Ring doorbell to initiate transfer */
	/* In practice, this would be done by a separate kernel or callback */
}

void nccl_ki_wait_counter(struct nccl_ki_comm *comm, int counter_idx,
			  uint64_t expected, cudaStream_t stream)
{
	/* Launch a kernel that polls the counter writeback */
	/* The counter writeback is in GPU-accessible memory */
	/* This is a simplified example - real impl would be more sophisticated */
}

/*
 * =============================================================================
 * Example: Complete Send/Recv Pattern
 * =============================================================================
 */

/*
 * GPU kernel that performs a send with completion signaling.
 *
 * Pattern:
 *   1. POST PUT with counter binding
 *   2. POST triggered signal (fires when PUT completes)
 *   3. Ring doorbell
 *   4. Wait for local counter (ensures our PUT completed)
 */
__global__ void ki_send_with_signal(fi_ki_meta_t meta, int peer_idx,
				    const void *src, uint64_t dst_off,
				    size_t len, uint32_t counter_idx,
				    uint32_t signal_idx)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	/* Post PUT with automatic signal on completion */
	fi_cxi_ki::inject_put(meta, 0, peer_idx, src, dst_off, len, signal_idx,
			      1 /* signal value */, counter_idx);

	/* Ring doorbell to start transfer */
	fi_cxi_ring_doorbell(meta, 0);

	/* Wait for our PUT to complete locally */
	fi_cxi_cntr_wait(meta, counter_idx, 1);
}

/*
 * GPU kernel that waits for incoming data via signal.
 */
__global__ void ki_recv_wait(fi_ki_meta_t meta, uint32_t signal_idx,
			     uint64_t expected)
{
	if (threadIdx.x != 0 || blockIdx.x != 0)
		return;

	/* Wait for sender's signal indicating data is ready */
	fi_cxi_signal_wait(meta, signal_idx, expected);

	/* Data is now available in our receive buffer */
}

/*
 * =============================================================================
 * Example: Using the Fluent Builder API
 * =============================================================================
 */

/*
 * Demonstrates the fluent builder API for complex command construction.
 */
__global__ void ki_fluent_api_example(fi_ki_meta_t meta, int peer_idx,
				      const void *src, uint64_t dst_off,
				      size_t len)
{
	if (threadIdx.x != 0)
		return;

	/* Fluent API for building PUT commands */
	fi_cxi_ki::put(meta, 0)
		.opcode(fi_cxi_ki::DmaOp::Put)
		.peer(peer_idx)
		.local_lac()
		.local_addr(src)
		.remote_offset(dst_off)
		.length(len)
		.restricted()
		.counter(0);

	/* Fluent API for building AMO commands */
	auto *amo_cmd = fi_cxi_ki::amo(meta, 0)
				.opcode(fi_cxi_ki::DmaOp::Atomic)
				.peer(peer_idx)
				.remote_offset(dst_off)
				.length(8)
				.restricted()
				.get();

	/* Manually set atomic-specific fields */
	amo_cmd->atomic_op = static_cast<uint8_t>(fi_cxi_ki::AmoOp::Sum);
	amo_cmd->atomic_type =
		static_cast<uint8_t>(fi_cxi_ki::AtomicType::Int64);
	amo_cmd->op2_word1 = 42; /* value to add */
}
