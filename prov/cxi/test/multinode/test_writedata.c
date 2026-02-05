/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

/**
 * Comprehensive multinode fi_writedata() test.
 *
 * This test validates all fi_writedata() and fi_inject_writedata() variants
 * across multiple nodes using the multinode framework.
 *
 * == Writedata Mode Coverage ==
 *
 * The CXI provider implements three writedata modes:
 *
 * 1. SINGLE_PUT mode:
 *    - Small messages (<= threshold) to standard MRs
 *    - Single unrestricted PUT carries data + header_data
 *    - Tested with: -m single_put or sizes <= 2048 (default threshold)
 *
 * 2. TRIGGERED mode:
 *    - Large messages with write counter bound
 *    - Data PUT increments CT, notification PUT is triggered
 *    - Tested with: -m triggered (requires counter binding)
 *
 * 3. FLUSH mode:
 *    - Large messages without write counter, or prefer_flush=1
 *    - Data PUT with flush, then notification PUT
 *    - Tested with: -m flush or FI_CXI_WRITEDATA_PREFER_FLUSH=1
 *
 * == Test Variants ==
 *
 * - With/without counter binding (affects TRIGGERED vs FLUSH selection)
 * - Standard MRs vs optimized MRs (optimized always use two-PUT)
 * - Various transfer sizes (below/at/above threshold)
 * - fi_writedata vs fi_inject_writedata
 * - Zero-byte transfers
 *
 * Usage: srun -N2 ./test_writedata [options]
 *
 * Options:
 *   -s SIZE    Transfer size in bytes (default: 64)
 *   -n COUNT   Number of iterations (default: 10)
 *   -m MODE    Force mode: single_put, triggered, flush, or all
 *   -t THRESH  Set atomic threshold (default: 2048)
 *   -C         Disable counter binding (forces FLUSH for large)
 *   -O         Use optimized MRs (always two-PUT)
 *   -i         Use fi_inject_writedata
 *   -z         Include zero-byte transfer test
 *   -a         Run all variant tests automatically
 *   -v         Verbose output
 *   -h         Show help
 */

#include "multinode_frmwk.h"
#include <assert.h>
#include <cxip.h>
#include <getopt.h>
#include <malloc.h>
#include <ofi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TRACE(fmt, ...) CXIP_COLL_TRACE(CXIP_TRC_TEST_CODE, fmt, ##__VA_ARGS__)

/* Test configuration */
typedef struct test_config {
	size_t xfer_size;
	int num_iterations;
	bool use_inject;
	bool use_optimized_mr;
	bool disable_counter;
	bool test_zero_byte;
	bool run_all_variants;
	size_t atomic_threshold;
	const char *force_mode; /* "single_put", "triggered", "flush", "all" */
	bool verbose;
} test_config_t;

static test_config_t cfg = {
	.xfer_size = 64,
	.num_iterations = 10,
	.use_inject = false,
	.use_optimized_mr = false,
	.disable_counter = false,
	.test_zero_byte = false,
	.run_all_variants = false,
	.atomic_threshold = 2048,
	.force_mode = NULL,
	.verbose = false,
};

/* Memory region for receive buffer */
struct mem_region {
	uint8_t *mem;
	struct fid_mr *mr;
	uint64_t key;
};

/* Test context - resources for one test run */
typedef struct test_ctx {
	struct mem_region recv_mr;
	uint8_t *send_buf;
	size_t buf_size;
	fi_addr_t *fiaddrs;
	size_t num_fiaddrs;
	uint64_t *remote_keys; /* Array of remote MR keys from all ranks */
	bool counter_bound;
	bool optimized_mr;
} test_ctx_t;

#define WRITEDATA_MR_KEY_BASE 0x1000

/* Immediate data encodes sender rank, iteration, and test_id */
#define MAKE_IMM_DATA(rank, iter, test_id)                          \
	(((uint64_t) (rank) << 48) | ((uint64_t) (test_id) << 32) | \
	 ((uint64_t) (iter) & 0xFFFFFFFF))
#define IMM_DATA_RANK(data) ((int) ((data) >> 48))
#define IMM_DATA_TEST(data) ((int) (((data) >> 32) & 0xFFFF))
#define IMM_DATA_ITER(data) ((int) ((data) & 0xFFFFFFFF))

/* Test IDs for tracking */
enum test_id {
	TEST_RING_BASIC = 1,
	TEST_RING_LARGE,
	TEST_RING_THRESHOLD,
	TEST_RING_ZERO_BYTE,
	TEST_ALLTOALL,
	TEST_MODE_SINGLE_PUT,
	TEST_MODE_TRIGGERED,
	TEST_MODE_FLUSH,
	TEST_OPTIMIZED_MR,
	TEST_NO_COUNTER,
};

static const char *test_id_name(enum test_id id)
{
	switch (id) {
	case TEST_RING_BASIC:
		return "ring_basic";
	case TEST_RING_LARGE:
		return "ring_large";
	case TEST_RING_THRESHOLD:
		return "ring_threshold";
	case TEST_RING_ZERO_BYTE:
		return "ring_zero_byte";
	case TEST_ALLTOALL:
		return "alltoall";
	case TEST_MODE_SINGLE_PUT:
		return "mode_single_put";
	case TEST_MODE_TRIGGERED:
		return "mode_triggered";
	case TEST_MODE_FLUSH:
		return "mode_flush";
	case TEST_OPTIMIZED_MR:
		return "optimized_mr";
	case TEST_NO_COUNTER:
		return "no_counter";
	default:
		return "unknown";
	}
}

static void usage(const char *prog)
{
	fprintf(stderr, "Usage: %s [options]\n", prog);
	fprintf(stderr, "\nOptions:\n");
	fprintf(stderr, "  -s SIZE    Transfer size in bytes (default: 64)\n");
	fprintf(stderr, "  -n COUNT   Number of iterations (default: 10)\n");
	fprintf(stderr, "  -m MODE    Force mode: single_put, triggered, "
			"flush, or all\n");
	fprintf(stderr, "  -t THRESH  Set atomic threshold (default: 2048)\n");
	fprintf(stderr, "  -C         Disable counter binding (forces FLUSH "
			"for large)\n");
	fprintf(stderr, "  -O         Use optimized MRs (always two-PUT)\n");
	fprintf(stderr, "  -i         Use fi_inject_writedata\n");
	fprintf(stderr, "  -z         Include zero-byte transfer test\n");
	fprintf(stderr, "  -a         Run all variant tests automatically\n");
	fprintf(stderr, "  -v         Verbose output\n");
	fprintf(stderr, "  -h         Show help\n");
	fprintf(stderr, "\nExamples:\n");
	fprintf(stderr,
		"  srun -N2 ./test_writedata -a          # Run all tests\n");
	fprintf(stderr,
		"  srun -N2 ./test_writedata -m flush    # Test FLUSH mode\n");
	fprintf(stderr, "  srun -N2 ./test_writedata -s 8192     # Test large "
			"transfers\n");
	fprintf(stderr, "  srun -N2 ./test_writedata -O          # Test "
			"optimized MRs\n");
	fprintf(stderr, "  srun -N2 ./test_writedata -C -s 4096  # Large "
			"without counter\n");
}

static int parse_args(int argc, char **argv)
{
	int opt;

	while ((opt = getopt(argc, argv, "s:n:m:t:COizavh")) != -1) {
		switch (opt) {
		case 's':
			cfg.xfer_size = (size_t) atol(optarg);
			break;
		case 'n':
			cfg.num_iterations = atoi(optarg);
			break;
		case 'm':
			cfg.force_mode = optarg;
			break;
		case 't':
			cfg.atomic_threshold = (size_t) atol(optarg);
			break;
		case 'C':
			cfg.disable_counter = true;
			break;
		case 'O':
			cfg.use_optimized_mr = true;
			break;
		case 'i':
			cfg.use_inject = true;
			break;
		case 'z':
			cfg.test_zero_byte = true;
			break;
		case 'a':
			cfg.run_all_variants = true;
			break;
		case 'v':
			cfg.verbose = true;
			break;
		case 'h':
		default:
			usage(argv[0]);
			return -1;
		}
	}

	/* Inject has size limits */
	if (cfg.use_inject && cfg.xfer_size > 64) {
		frmwk_log0(
			"Warning: inject limited to 64 bytes, reducing size\n");
		cfg.xfer_size = 64;
	}

	return 0;
}

/**
 * Set environment variable for threshold.
 */
static void set_threshold_env(size_t threshold)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%zu", threshold);
	setenv("FI_CXI_WRITEDATA_ATOMIC_THRESHOLD", buf, 1);
}

/**
 * Set prefer_flush environment variable.
 */
static void set_prefer_flush_env(bool prefer)
{
	setenv("FI_CXI_WRITEDATA_PREFER_FLUSH", prefer ? "1" : "0", 1);
}

/**
 * Create a memory region for receiving writedata.
 *
 * @param len       Buffer length
 * @param key       MR key
 * @param bind_cntr Whether to bind remote counter
 * @param optimized Whether to use optimized MR (provider key)
 * @param mr        Output memory region
 */
static int create_recv_mr(size_t len, uint64_t key, bool bind_cntr,
			  bool optimized, struct mem_region *mr)
{
	struct fi_mr_attr mr_attr = {};
	struct iovec iov;
	uint64_t flags = 0;
	int ret;

	mr->mem = calloc(1, len ? len : 1);
	if (!mr->mem) {
		frmwk_log("Failed to allocate recv buffer\n");
		return -FI_ENOMEM;
	}
	mr->key = key;

	/* fi_writedata requires:
	 * 1. Provider keys (FI_MR_PROV_KEY) - client keys don't encode events
	 * 2. FI_RMA_EVENT flag (in flags, not access) - enables remote CQ data
	 * delivery
	 *
	 * The 'optimized' flag only affects whether to use optimized MR mode,
	 * but all writedata MRs need provider keys and RMA events.
	 */
	iov.iov_base = mr->mem;
	iov.iov_len = len ? len : 1;
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_REMOTE_WRITE;
	mr_attr.offset = 0;
	mr_attr.requested_key = key;
	mr_attr.context = NULL;
	flags = FI_MR_PROV_KEY | FI_RMA_EVENT;

	ret = fi_mr_regattr(cxit_domain, &mr_attr, flags, &mr->mr);

	if (ret) {
		frmwk_log("fi_mr_regattr failed: %d\n", ret);
		free(mr->mem);
		return ret;
	}

	ret = fi_mr_bind(mr->mr, &cxit_ep->fid, 0);
	if (ret) {
		frmwk_log("fi_mr_bind(ep) failed: %d\n", ret);
		fi_close(&mr->mr->fid);
		free(mr->mem);
		return ret;
	}

	/* Bind to remote counter for FI_RMA_EVENT if requested */
	if (bind_cntr && cxit_rem_cntr) {
		ret = fi_mr_bind(mr->mr, &cxit_rem_cntr->fid, FI_REMOTE_WRITE);
		if (ret) {
			frmwk_log("fi_mr_bind(cntr) failed: %d\n", ret);
			fi_close(&mr->mr->fid);
			free(mr->mem);
			return ret;
		}
	}

	ret = fi_mr_enable(mr->mr);
	if (ret) {
		frmwk_log("fi_mr_enable failed: %d\n", ret);
		fi_close(&mr->mr->fid);
		free(mr->mem);
		return ret;
	}

	/* Get actual key (may differ for optimized MRs) */
	mr->key = fi_mr_key(mr->mr);

	return 0;
}

static void destroy_recv_mr(struct mem_region *mr)
{
	if (mr->mr)
		fi_close(&mr->mr->fid);
	free(mr->mem);
	mr->mr = NULL;
	mr->mem = NULL;
}

/**
 * Wait for TX completion.
 */
static int wait_tx_cq(void)
{
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err;
	ssize_t ret;

	do {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		if (ret == -FI_EAVAIL) {
			fi_cq_readerr(cxit_tx_cq, &err, 0);
			frmwk_log("TX CQ error: %s\n",
				  fi_cq_strerror(cxit_tx_cq, err.prov_errno,
						 err.err_data, NULL, 0));
			return -err.err;
		}
	} while (ret == -FI_EAGAIN);

	if (ret < 0) {
		frmwk_log("fi_cq_read(tx) failed: %zd\n", ret);
		return ret;
	}

	/* Verify TX completion flags */
	if (!(cqe.flags & FI_RMA) || !(cqe.flags & FI_WRITE)) {
		frmwk_log("Unexpected TX flags: %#llx\n",
			  (unsigned long long) cqe.flags);
		return -FI_EINVAL;
	}

	return 0;
}

/**
 * Wait for RX completion (remote CQ data event).
 */
static int wait_rx_cq(uint64_t *imm_data, size_t *len)
{
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err;
	ssize_t ret;

	do {
		ret = fi_cq_read(cxit_rx_cq, &cqe, 1);
		if (ret == -FI_EAVAIL) {
			fi_cq_readerr(cxit_rx_cq, &err, 0);
			frmwk_log("RX CQ error: %s\n",
				  fi_cq_strerror(cxit_rx_cq, err.prov_errno,
						 err.err_data, NULL, 0));
			return -err.err;
		}
	} while (ret == -FI_EAGAIN);

	if (ret < 0) {
		frmwk_log("fi_cq_read(rx) failed: %zd\n", ret);
		return ret;
	}

	/* Verify RX completion has FI_REMOTE_CQ_DATA */
	if (!(cqe.flags & FI_REMOTE_CQ_DATA)) {
		frmwk_log("Missing FI_REMOTE_CQ_DATA flag: %#llx\n",
			  (unsigned long long) cqe.flags);
		return -FI_EINVAL;
	}

	if (!(cqe.flags & FI_RMA) || !(cqe.flags & FI_REMOTE_WRITE)) {
		frmwk_log("Missing RMA flags: %#llx\n",
			  (unsigned long long) cqe.flags);
		return -FI_EINVAL;
	}

	*imm_data = cqe.data;
	if (len)
		*len = cqe.len;
	return 0;
}

/**
 * Initialize test context.
 */
static int init_test_ctx(test_ctx_t *ctx, size_t buf_size, uint64_t mr_key,
			 bool bind_cntr, bool optimized)
{
	int ret;

	/* Note: Do NOT memset the whole ctx here - fiaddrs/num_fiaddrs are
	 * already populated by init_libfabric_and_av() and must be preserved.
	 */
	memset(&ctx->recv_mr, 0, sizeof(ctx->recv_mr));
	ctx->send_buf = NULL;
	ctx->remote_keys = NULL;
	ctx->buf_size = buf_size;
	ctx->counter_bound = bind_cntr;
	ctx->optimized_mr = optimized;

	/* Allocate send buffer */
	ctx->send_buf = malloc(buf_size ? buf_size : 1);
	if (!ctx->send_buf) {
		frmwk_log("Failed to allocate send buffer\n");
		return -FI_ENOMEM;
	}

	/* Create receive MR */
	ret = create_recv_mr(buf_size, mr_key, bind_cntr, optimized,
			     &ctx->recv_mr);
	if (ret) {
		frmwk_log("create_recv_mr failed: %d\n", ret);
		free(ctx->send_buf);
		return ret;
	}

	/* Exchange MR keys with all ranks.
	 * With provider keys (FI_MR_PROV_KEY), each rank gets a unique key
	 * from the provider. We need to know the remote rank's key to
	 * target their MR.
	 */
	ctx->remote_keys = calloc(frmwk_numranks, sizeof(uint64_t));
	if (!ctx->remote_keys) {
		frmwk_log("Failed to allocate remote_keys\n");
		destroy_recv_mr(&ctx->recv_mr);
		free(ctx->send_buf);
		return -FI_ENOMEM;
	}

	ret = frmwk_allgather(sizeof(uint64_t), &ctx->recv_mr.key,
			      ctx->remote_keys);
	if (ret) {
		frmwk_log("frmwk_allgather for keys failed: %d\n", ret);
		free(ctx->remote_keys);
		destroy_recv_mr(&ctx->recv_mr);
		free(ctx->send_buf);
		return ret;
	}

	if (cfg.verbose) {
		frmwk_log("Local MR key: 0x%lx\n", ctx->recv_mr.key);
		for (int i = 0; i < frmwk_numranks; i++)
			frmwk_log("  Rank %d key: 0x%lx\n", i,
				  ctx->remote_keys[i]);
	}

	return 0;
}

static void cleanup_test_ctx(test_ctx_t *ctx)
{
	destroy_recv_mr(&ctx->recv_mr);
	free(ctx->send_buf);
	ctx->send_buf = NULL;
	free(ctx->remote_keys);
	ctx->remote_keys = NULL;
}

/**
 * Run a single writedata operation and verify.
 */
static int do_writedata(test_ctx_t *ctx, fi_addr_t dest_addr, uint64_t dest_key,
			size_t len, uint64_t imm_data, bool use_inject)
{
	int ret;

	/* Fill send buffer with pattern based on immediate data */
	for (size_t i = 0; i < len; i++)
		ctx->send_buf[i] = (uint8_t) ((imm_data + i) & 0xff);

	if (use_inject) {
		ret = fi_inject_writedata(cxit_ep, ctx->send_buf, len, imm_data,
					  dest_addr, 0, dest_key);
	} else {
		ret = fi_writedata(cxit_ep, ctx->send_buf, len, NULL, imm_data,
				   dest_addr, 0, dest_key, NULL);
	}

	if (ret) {
		frmwk_log("fi_%swritedata failed: %d\n",
			  use_inject ? "inject_" : "", ret);
		return ret;
	}

	/* Wait for TX completion (only for non-inject) */
	if (!use_inject) {
		ret = wait_tx_cq();
		if (ret) {
			frmwk_log("TX completion failed: %d\n", ret);
			return ret;
		}
	}

	return 0;
}

/**
 * Verify received data.
 */
static int verify_recv(test_ctx_t *ctx, uint64_t expected_imm,
		       int expected_src_rank, size_t expected_len)
{
	uint64_t recv_imm;
	size_t recv_len;
	int ret;

	ret = wait_rx_cq(&recv_imm, &recv_len);
	if (ret) {
		frmwk_log("RX completion failed: %d\n", ret);
		return ret;
	}

	/* Verify immediate data source rank */
	if (IMM_DATA_RANK(recv_imm) != expected_src_rank) {
		frmwk_log("Immediate data rank mismatch: expected %d, got %d\n",
			  expected_src_rank, IMM_DATA_RANK(recv_imm));
		return -FI_EINVAL;
	}

	/* Verify data pattern - use volatile to prevent compiler caching reads
	 */
	volatile uint8_t *recv_buf = (volatile uint8_t *) ctx->recv_mr.mem;

	for (size_t i = 0; i < expected_len; i++) {
		uint8_t expected = (uint8_t) ((expected_imm + i) & 0xff);
		uint8_t actual = recv_buf[i];

		if (actual != expected) {
			frmwk_log("Data mismatch at offset %zu: "
				  "expected %#x, got %#x (recv_imm=%#lx "
				  "expected_imm=%#lx len=%zu)\n",
				  i, expected, actual, (unsigned long) recv_imm,
				  (unsigned long) expected_imm, expected_len);
			/* Dump first few bytes */
			frmwk_log("Buffer dump: ");
			for (size_t j = 0;
			     j < (expected_len < 16 ? expected_len : 16); j++)
				fprintf(stderr, "%02x ", recv_buf[j]);
			fprintf(stderr, "\n");
			return -FI_EINVAL;
		}
	}

	return 0;
}

/**
 * Run ring test with specified parameters.
 */
static int run_ring_test(test_ctx_t *ctx, enum test_id test_id,
			 size_t xfer_size, int iterations, bool use_inject)
{
	int dest_rank = (frmwk_rank + 1) % frmwk_numranks;
	int src_rank = (frmwk_rank - 1 + frmwk_numranks) % frmwk_numranks;
	fi_addr_t dest_addr = ctx->fiaddrs[dest_rank];
	uint64_t dest_key = ctx->remote_keys[dest_rank]; /* Use exchanged key */
	struct timespec t0, t1;
	int errors = 0;

	frmwk_log0("[%s] Ring test: %zu bytes, %d iterations, %s%s%s\n",
		   test_id_name(test_id), xfer_size, iterations,
		   use_inject ? "inject" : "writedata",
		   ctx->optimized_mr ? ", optimized MR" : "",
		   ctx->counter_bound ? "" : ", no counter");

	clock_gettime(CLOCK_MONOTONIC, &t0);

	for (int iter = 0; iter < iterations; iter++) {
		uint64_t imm_send = MAKE_IMM_DATA(frmwk_rank, iter, test_id);
		uint64_t imm_expected = MAKE_IMM_DATA(src_rank, iter, test_id);
		int ret;

		/* Clear receive buffer */
		memset(ctx->recv_mr.mem, 0, ctx->buf_size ? ctx->buf_size : 1);

		/* Barrier to synchronize */
		ret = frmwk_barrier();
		if (ret) {
			frmwk_log("Barrier failed: %d\n", ret);
			return ret;
		}

		/* Send writedata */
		ret = do_writedata(ctx, dest_addr, dest_key, xfer_size,
				   imm_send, use_inject);
		if (ret) {
			errors++;
			continue;
		}

		/* Verify receive */
		ret = verify_recv(ctx, imm_expected, src_rank, xfer_size);
		if (ret) {
			errors++;
			continue;
		}

		if (cfg.verbose) {
			frmwk_log("Iteration %d: OK\n", iter);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &t1);

	/* Calculate elapsed time */
	if (t1.tv_nsec < t0.tv_nsec) {
		t1.tv_nsec += 1000000000;
		t1.tv_sec--;
	}
	t1.tv_nsec -= t0.tv_nsec;
	t1.tv_sec -= t0.tv_sec;

	frmwk_barrier();

	frmwk_log0("[%s] Completed in %ld.%09ld sec, %d errors\n",
		   test_id_name(test_id), t1.tv_sec, t1.tv_nsec, errors);

	return errors ? -FI_EIO : 0;
}

/**
 * Initialize libfabric and populate AV.
 */
static int init_libfabric_and_av(test_ctx_t *ctx)
{
	int ret;

	/* Configure framework for writedata requirements:
	 * - FI_RMA: Required for RMA operations
	 * - FI_RMA_EVENT: Required for writedata remote CQ
	 * notifications
	 * - FI_MR_ALLOCATED | FI_MR_PROV_KEY: Required for writedata MR
	 * encoding
	 */
	frmwk_extra_caps = FI_RMA | FI_RMA_EVENT;
	frmwk_extra_mr_mode = FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	ret = frmwk_init_libfabric();
	if (ret) {
		frmwk_log("frmwk_init_libfabric failed: %d\n", ret);
		return ret;
	}

	ret = frmwk_populate_av(&ctx->fiaddrs, &ctx->num_fiaddrs);
	if (ret) {
		frmwk_log("frmwk_populate_av failed: %d\n", ret);
		frmwk_free_libfabric();
		return ret;
	}

	return 0;
}

static void cleanup_libfabric_and_av(test_ctx_t *ctx)
{
	free(ctx->fiaddrs);
	ctx->fiaddrs = NULL;
	frmwk_free_libfabric();
}

/*
 * =============================================================================
 * Individual test variants
 * =============================================================================
 */

/**
 * Test 1: Basic small transfer (SINGLE_PUT mode expected).
 */
static int test_basic_small(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Basic Small Transfer (SINGLE_PUT mode) "
		   "===\n");

	/* Set threshold to ensure small messages use SINGLE_PUT */
	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, 64, WRITEDATA_MR_KEY_BASE, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_MODE_SINGLE_PUT, 64, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 2: Large transfer with counter (TRIGGERED mode expected).
 */
static int test_large_with_counter(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Large Transfer with Counter (TRIGGERED "
		   "mode) "
		   "===\n");

	/* Set threshold below transfer size */
	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	/* With counter binding, large transfers should use TRIGGERED
	 * mode */
	ret = init_test_ctx(&ctx, 8192, WRITEDATA_MR_KEY_BASE + 1, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_MODE_TRIGGERED, 8192, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 3: Large transfer without counter (FLUSH mode expected).
 */
static int test_large_no_counter(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Large Transfer without Counter (FLUSH "
		   "mode) "
		   "===\n");

	/* Set threshold below transfer size */
	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	/* Without counter binding, large transfers should use FLUSH
	 * mode */
	ret = init_test_ctx(&ctx, 8192, WRITEDATA_MR_KEY_BASE + 2, false,
			    false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_NO_COUNTER, 8192, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 4: Forced FLUSH mode via environment variable.
 */
static int test_prefer_flush(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Forced FLUSH Mode (prefer_flush=1) ===\n");

	/* Force FLUSH mode even with counter */
	set_threshold_env(2048);
	set_prefer_flush_env(true);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, 8192, WRITEDATA_MR_KEY_BASE + 3, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_MODE_FLUSH, 8192, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	/* Reset prefer_flush */
	set_prefer_flush_env(false);

	return ret;
}

/**
 * Test 5: Optimized MR (always two-PUT, never SINGLE_PUT).
 */
static int test_optimized_mr(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Optimized MR (always two-PUT) ===\n");

	/* Even small messages should use two-PUT with optimized MR */
	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	/* Optimized MR with counter binding */
	ret = init_test_ctx(&ctx, 64, WRITEDATA_MR_KEY_BASE + 4, true, true);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_OPTIMIZED_MR, 64, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 6: Threshold boundary test (at and above threshold).
 */
static int test_threshold_boundary(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Threshold Boundary ===\n");

	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	/* Need buffer large enough for boundary tests */
	ret = init_test_ctx(&ctx, 4096, WRITEDATA_MR_KEY_BASE + 5, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	/* Test exactly at threshold (should use SINGLE_PUT) */
	frmwk_log0("  Testing at threshold (2048 bytes)...\n");
	ret = run_ring_test(&ctx, TEST_RING_THRESHOLD, 2048, 3, false);
	if (ret) {
		cleanup_test_ctx(&ctx);
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	/* Test just above threshold (should use two-PUT) */
	frmwk_log0("  Testing above threshold (2049 bytes)...\n");
	ret = run_ring_test(&ctx, TEST_RING_THRESHOLD, 2049, 3, false);
	if (ret) {
		cleanup_test_ctx(&ctx);
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return 0;
}

/**
 * Test 7: Zero threshold (always two-PUT).
 */
static int test_zero_threshold(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Zero Threshold (always two-PUT) ===\n");

	/* Threshold=0 disables SINGLE_PUT optimization */
	set_threshold_env(0);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, 64, WRITEDATA_MR_KEY_BASE + 6, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	/* Small message should still use two-PUT with threshold=0 */
	ret = run_ring_test(&ctx, TEST_RING_BASIC, 64, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	/* Reset threshold */
	set_threshold_env(2048);

	return ret;
}

/**
 * Test 8: Zero-byte transfer.
 */
static int test_zero_byte(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: Zero-Byte Transfer ===\n");

	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, 0, WRITEDATA_MR_KEY_BASE + 7, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_RING_ZERO_BYTE, 0, cfg.num_iterations,
			    false);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 9: fi_inject_writedata.
 */
static int test_inject_writedata(void)
{
	test_ctx_t ctx;
	int ret;

	frmwk_log0("\n=== Test: fi_inject_writedata ===\n");

	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, 64, WRITEDATA_MR_KEY_BASE + 8, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_RING_BASIC, 64, cfg.num_iterations,
			    true);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Test 10: Multiple sizes in sequence.
 */
static int test_multiple_sizes(void)
{
	test_ctx_t ctx;
	size_t sizes[] = {1, 64, 512, 2048, 4096, 8192};
	int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
	int ret;

	frmwk_log0("\n=== Test: Multiple Sizes ===\n");

	set_threshold_env(2048);
	set_prefer_flush_env(false);

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	/* Create large buffer for all sizes */
	ret = init_test_ctx(&ctx, 8192, WRITEDATA_MR_KEY_BASE + 9, true, false);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	for (int i = 0; i < num_sizes; i++) {
		frmwk_log0("  Testing size %zu...\n", sizes[i]);
		ret = run_ring_test(&ctx, TEST_RING_BASIC, sizes[i], 3, false);
		if (ret) {
			frmwk_log0("  Size %zu FAILED\n", sizes[i]);
			break;
		}
	}

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

/**
 * Run all variant tests.
 */
static int run_all_tests(void)
{
	int ret;
	int failures = 0;

	frmwk_log0("\n========================================\n");
	frmwk_log0("Running All Writedata Variant Tests\n");
	frmwk_log0("========================================\n");

	/* Test 1: Basic small (SINGLE_PUT) */
	ret = test_basic_small();
	if (ret) {
		frmwk_log0("FAILED: test_basic_small\n");
		failures++;
	}

	/* Test 2: Large with counter (TRIGGERED) */
	ret = test_large_with_counter();
	if (ret) {
		frmwk_log0("FAILED: test_large_with_counter\n");
		failures++;
	}

	/* Test 3: Large without counter (FLUSH) */
	ret = test_large_no_counter();
	if (ret) {
		frmwk_log0("FAILED: test_large_no_counter\n");
		failures++;
	}

	/* Test 4: Prefer flush env var */
	ret = test_prefer_flush();
	if (ret) {
		frmwk_log0("FAILED: test_prefer_flush\n");
		failures++;
	}

	/* Test 5: Optimized MR */
	ret = test_optimized_mr();
	if (ret) {
		frmwk_log0("FAILED: test_optimized_mr\n");
		failures++;
	}

	/* Test 6: Threshold boundary */
	ret = test_threshold_boundary();
	if (ret) {
		frmwk_log0("FAILED: test_threshold_boundary\n");
		failures++;
	}

	/* Test 7: Zero threshold */
	ret = test_zero_threshold();
	if (ret) {
		frmwk_log0("FAILED: test_zero_threshold\n");
		failures++;
	}

	/* Test 8: Zero-byte */
	if (cfg.test_zero_byte) {
		ret = test_zero_byte();
		if (ret) {
			frmwk_log0("FAILED: test_zero_byte\n");
			failures++;
		}
	}

	/* Test 9: Inject writedata */
	ret = test_inject_writedata();
	if (ret) {
		frmwk_log0("FAILED: test_inject_writedata\n");
		failures++;
	}

	/* Test 10: Multiple sizes */
	ret = test_multiple_sizes();
	if (ret) {
		frmwk_log0("FAILED: test_multiple_sizes\n");
		failures++;
	}

	frmwk_log0("\n========================================\n");
	if (failures == 0) {
		frmwk_log0("All tests PASSED\n");
	} else {
		frmwk_log0("%d test(s) FAILED\n", failures);
	}
	frmwk_log0("========================================\n");

	return failures ? -1 : 0;
}

/**
 * Run single mode test based on command line.
 */
static int run_single_test(void)
{
	test_ctx_t ctx;
	bool use_counter = !cfg.disable_counter;
	int ret;

	/* Set environment */
	set_threshold_env(cfg.atomic_threshold);

	if (cfg.force_mode && strcmp(cfg.force_mode, "flush") == 0) {
		set_prefer_flush_env(true);
	} else {
		set_prefer_flush_env(false);
	}

	ret = init_libfabric_and_av(&ctx);
	if (ret)
		return ret;

	ret = init_test_ctx(&ctx, cfg.xfer_size, WRITEDATA_MR_KEY_BASE,
			    use_counter, cfg.use_optimized_mr);
	if (ret) {
		cleanup_libfabric_and_av(&ctx);
		return ret;
	}

	ret = run_ring_test(&ctx, TEST_RING_BASIC, cfg.xfer_size,
			    cfg.num_iterations, cfg.use_inject);

	cleanup_test_ctx(&ctx);
	cleanup_libfabric_and_av(&ctx);

	return ret;
}

int main(int argc, char **argv)
{
	int ret;

	/* Parse command line arguments */
	if (parse_args(argc, argv))
		return 1;

	/* Initialize framework */
	frmwk_init(false);

	/* Check minimum ranks */
	ret = frmwk_check_env(2);
	if (ret) {
		frmwk_log0("Need at least 2 ranks for writedata test\n");
		frmwk_term();
		return 1;
	}

	/* Gather NIC addresses */
	ret = frmwk_gather_nics();
	if (ret) {
		frmwk_log0("frmwk_gather_nics failed: %d\n", ret);
		frmwk_term();
		return 1;
	}

	frmwk_log0("Initialized: rank %d of %d\n", frmwk_rank, frmwk_numranks);

	if (cfg.run_all_variants) {
		ret = run_all_tests();
	} else {
		ret = run_single_test();
	}

	frmwk_term();

	return ret ? 1 : 0;
}
