# KI API v4: Libfabric Host-Side Interface

This document describes the host-side libfabric CXI provider interface for KI resource setup. For GPU-side usage, see [GPU API](fi_cxi_ki_gpu.md). For lower-level implementation, see [libcxi API](fi_cxi_ki_libcxi.md) and [CXI Driver](fi_cxi_ki_driver.md).

---

## Summary: What's New

### New Extension: `fi_cxi_ki_ops`

```c
#define FI_CXI_KI_OPS_1 "ki_ops_v1"

struct fi_cxi_ki_ops {
    int (*get_cmdq_info)(...);    // Requires new libcxi accessor
    int (*get_ep_info)(...);      // Uses existing provider data
    int (*resolve_target)(...);   // Uses existing AV lookup
    int (*get_mr_info)(...);      // Uses existing MR data
    int (*sync_cmdq_wp)(...);     // Post-kernel write pointer sync
};
```

### New: Metadata Structure

`struct fi_cxi_ki_meta` packages all KI resources for GPU kernel access using SoA layout for coalesced memory access.

### Dependencies

| `fi_cxi_ki_ops` Method | Data Source |
|------------------------|-------------|
| `get_cmdq_info` | Existing `struct cxi_cq` fields: `cmds`, `size`, `wp_addr` |
| `get_ep_info` | Existing provider data |
| `resolve_target` | Existing `cxip_av_lookup_addr()` |
| `get_mr_info` | Existing `struct cxi_md` fields: `iova`, `lac` |

No libcxi changes required — structures are already public.

---

## Existing Provider Extensions

The CXI provider already exposes several extension APIs that KI builds upon.

### Counter Operations (fi_cxi_cntr_ops)

The existing counter operations extension provides GPU-accessible counter access:

```c
/* From fi_cxi_ext.h */
#define FI_CXI_COUNTER_OPS "cxi_counter_ops"

struct fi_cxi_cntr_ops {
    /* Set the counter writeback address to a client provided address. */
    int (*set_wb_buffer)(struct fid *fid, void *buf, size_t len);

    /* Get the counter MMIO region (doorbell). */
    int (*get_mmio_addr)(struct fid *fid, void **addr, size_t *len);
};
```

**Usage:**
```c
struct fi_cxi_cntr_ops *cntr_ops;
fi_open_ops(&cntr->fid, FI_CXI_COUNTER_OPS, 0, (void **)&cntr_ops, NULL);

/* Set writeback to GPU memory */
void *gpu_wb_buf = /* GPU-allocated via cudaMalloc/hipMalloc */;
cntr_ops->set_wb_buffer(&cntr->fid, gpu_wb_buf, sizeof(struct c_ct_writeback));

/* Get doorbell for GPU writes */
void *mmio_addr;
size_t mmio_len;
cntr_ops->get_mmio_addr(&cntr->fid, &mmio_addr, &mmio_len);
```

**Counter MMIO Operations (from fi_cxi_ext.h):**

These values match the hardware `struct c_ct_writeback` structure (see [CXI Driver](fi_cxi_ki_driver.md#ct-writeback-structure)):

```c
/* Success values: 48-bit max (matches c_ct_writeback.ct_success field) */
#define FI_CXI_CNTR_SUCCESS_MAX ((1ULL << 48) - 1)
/* Failure values: 7-bit max (matches c_ct_writeback.ct_failure field) */
#define FI_CXI_CNTR_FAILURE_MAX ((1ULL << 7) - 1)

/* Read from writeback buffer (interprets c_ct_writeback structure) */
uint64_t fi_cxi_cntr_wb_read(const void *wb_buf);
uint64_t fi_cxi_cntr_wb_readerr(const void *wb_buf);

/* Write to MMIO doorbell region (wraps cxi_ct_inc_success/failure) */
int fi_cxi_cntr_add(void *cntr_mmio, uint64_t value);
int fi_cxi_cntr_adderr(void *cntr_mmio, uint64_t value);
int fi_cxi_cntr_set(void *cntr_mmio, uint64_t value);

/* Generate pollable value for GPU kernel */
int fi_cxi_gen_cntr_success(uint64_t value, uint64_t *cxi_value);
```

### Domain Operations (fi_cxi_dom_ops)

```c
/* From fi_cxi_ext.h */
#define FI_CXI_DOM_OPS_6 "dom_ops_v6"

struct fi_cxi_dom_ops {
    /* Read hardware performance counters */
    int (*cntr_read)(struct fid *fid, unsigned int cntr, uint64_t *value,
                     struct timespec *ts);

    /* Get network topology information */
    int (*topology)(struct fid *fid, unsigned int *group_id,
                    unsigned int *switch_id, unsigned int *port_id);

    /* Enable hybrid MR desc mode */
    int (*enable_hybrid_mr_desc)(struct fid *fid, bool enable);

    /* Get unexpected message information */
    size_t (*ep_get_unexp_msgs)(struct fid_ep *fid_ep, ...);

    /* Get deferred work queue depth */
    int (*get_dwq_depth)(struct fid *fid, size_t *depth);

    /* MR event control (deprecated, use fi_control) */
    int (*enable_mr_match_events)(struct fid *fid, bool enable);
    int (*enable_optimized_mrs)(struct fid *fid, bool enable);
};
```

---

## KI Operations Extension

KI adds a new operations structure following the existing pattern:

```c
#define FI_CXI_KI_OPS_1 "ki_ops_v1"

struct fi_cxi_ki_ops {
    /* Get command queue info for GPU command emission */
    int (*get_cmdq_info)(struct fid *fid, int tx_ctx_index,
                         struct fi_cxi_ki_cmdq_info *info);

    /* Get endpoint addressing info */
    int (*get_ep_info)(struct fid *fid, struct fi_cxi_ki_ep_info *info);

    /* Resolve peer fabric address to NIC routing info */
    int (*resolve_target)(struct fid *fid, fi_addr_t addr,
                          struct fi_cxi_ki_target_info *info);

    /* Get MR info for remote access */
    int (*get_mr_info)(struct fid *fid, struct fi_cxi_ki_mr_info *info);

    /* Sync write pointer after GPU kernel execution */
    int (*sync_cmdq_wp)(struct fid *fid, int tx_ctx_index, uint32_t num_cmds);

    /* Batch-allocate counters with contiguous GPU writeback buffers.
     *
     * This is REQUIRED for KI - individual fi_cntr_open() calls result in
     * non-contiguous writeback buffers which defeat GPU memory coalescing.
     *
     * The caller provides a pre-allocated contiguous GPU buffer. The provider
     * allocates 'count' counters, each with its writeback at:
     *   gpu_wb_base + i * sizeof(struct c_ct_writeback)
     *
     * Returns array of counter fids that must be closed individually.
     */
    int (*alloc_counters_batch)(struct fid *fid,
                                uint32_t count,
                                void *gpu_wb_base,          /* GPU memory */
                                struct fid_cntr **cntrs,    /* [count] output */
                                struct fi_cxi_ki_cntr_batch_info *info);

    /* Batch-allocate target CTs for receive counting (Model A).
     *
     * Similar to alloc_counters_batch but for target-side CTs.
     * These CTs are bound to PTEs to count incoming operations.
     */
    int (*alloc_target_cts_batch)(struct fid *fid,
                                  uint32_t count,
                                  void *gpu_wb_base,
                                  struct fi_cxi_ki_target_ct_info *info);
};

/* Info returned from batch counter allocation */
struct fi_cxi_ki_cntr_batch_info {
    void    *mmio_base;       /* Base doorbell address */
    size_t   mmio_stride;     /* Stride between doorbells (typically 4KB) */
    uint16_t *ct_indices;     /* [count] CT indices for command binding */
};

/* Info returned from batch target CT allocation */
struct fi_cxi_ki_target_ct_info {
    uint16_t *ct_indices;     /* [count] CT indices */
    uint32_t  pte_index;      /* PTE index for receive binding */
};
```

---

## Data Structures

### Command Queue Info

Maps to driver's `cxi_cq_alloc_resp` (see [CXI Driver](fi_cxi_ki_driver.md#cq-allocation)):

```c
struct fi_cxi_ki_cmdq_info {
    void     *cmdq_buf;      /* Command buffer base (from cxi_cq.cmds32) */
    uint32_t  cmdq_size;     /* Number of 32-byte command slots */
    uint32_t  cmdq_mask;     /* Wrap mask (size - 1) */
    uint32_t  cmd_slot_size; /* Bytes per base slot (always 32) */
    volatile uint64_t *wp_addr; /* Write pointer CSR (from cxi_cq.wp_addr) */
};
```

**Command Sizes:**
- Base slot: 32 bytes (`struct cxi_cmd32`)
- Standard DMA PUT: 64 bytes (2 slots)
- Full DMA with all options: 64 bytes (2 slots)
- Triggered operations: 128 bytes (4 slots)

The write pointer tracks 32-byte slots. A 64-byte command consumes 2 slots.

### Endpoint Info

```c
struct fi_cxi_ki_ep_info {
    uint32_t nid;            /* NIC address (C_DFA_NIC_BITS) */
    uint32_t pid;            /* Process ID (C_DFA_PID_BITS_MAX) */
    uint8_t  pid_bits;       /* PID bits for DFA construction */
    uint16_t vni;            /* Virtual Network ID */
};
```

### Target Info (Peer Resolution)

```c
struct fi_cxi_ki_target_info {
    /* NIC routing (populated by resolve_target) */
    uint64_t dfa;            /* Destination Fabric Address */
    uint8_t  dfa_ext;        /* DFA extension */
    uint32_t index_ext;      /* Index extension */
};
```

### Memory Registration Info

Maps to driver's `cxi_md` structure (see [CXI Driver](fi_cxi_ki_driver.md#7-memory-descriptor-md)):

```c
struct fi_cxi_ki_mr_info {
    uint64_t iova;           /* IO Virtual Address (from cxi_md.iova) */
    uint32_t lac;            /* Logical Access Context (from cxi_md.lac) */
    uint64_t key;            /* MR key for remote access */
};
```

---

## Metadata Structure (GPU-Side)

The metadata structure aggregates all KI resources for GPU kernel access.

### Design Principles

1. **Coalesced memory access**: Peer data uses Structure-of-Arrays (SoA) layout for GPU warp efficiency.
2. **Hot/warm/cold separation**: Frequently accessed data separated from rarely accessed.
3. **Cache-line alignment**: Write-heavy fields get dedicated cache lines.

### Structure Definition

```c
#define FI_KI_CACHE_LINE 128   /* GPU L2 cache line */
#define FI_KI_MAX_CONTEXTS 8   /* Max command contexts */

/* Command queue read-mostly config */
struct fi_cxi_ki_cmdq_config {
    void     *buf;           /* Command buffer base (32-byte slots) */
    uint32_t  mask;          /* Wrap mask (size - 1), in 32-byte slots */
    uint32_t  slot_size;     /* Bytes per base slot (always 32) */
};

/* Command queue write-heavy state (cache-line isolated) */
struct fi_cxi_ki_cmdq_hot {
    volatile uint64_t *wp_addr;   /* Write pointer CSR */
    volatile uint32_t  wp;        /* Local write pointer shadow */
    uint32_t           _pad[13];
} __attribute__((aligned(FI_KI_CACHE_LINE)));

/* Peer data - Structure of Arrays for coalesced GPU memory access.
 *
 * When 32 warp threads access consecutive peer indices, each array
 * generates a single coalesced memory transaction:
 *   dfa[base+0..31]       → 256 bytes (1-2 cache lines)
 *   dfa_ext[base+0..31]   → 32 bytes (1 cache line, partially used)
 *   index_ext[base+0..31] → 128 bytes (1 cache line)
 *   mr_base[base+0..31]   → 256 bytes (1-2 cache lines)
 *   mr_key[base+0..31]    → 256 bytes (1-2 cache lines)
 */
struct fi_cxi_ki_peer_soa {
    uint64_t *dfa;              /* [peer_count] Destination fabric addresses */
    uint8_t  *dfa_ext;          /* [peer_count] DFA extensions */
    uint32_t *index_ext;        /* [peer_count] Index extensions */
    uint64_t *mr_base;          /* [peer_count] Remote MR base IOVAs */
    uint64_t *mr_key;           /* [peer_count] Remote MR keys */
    uint64_t *signal_base;      /* [peer_count] Remote signal array base (Model B) */
    uint64_t *signal_key;       /* [peer_count] Remote signal MR keys (Model B) */
    uint32_t  count;            /* Number of peers */
};

/* Completion tracking - ALL use contiguous GPU memory for efficient access.
 *
 * KI REQUIREMENT: All writeback buffers must be allocated contiguously.
 * Use fi_cxi_ki_ops.alloc_counters_batch() to allocate counters with
 * contiguous GPU writeback buffers. This enables direct array access
 * without pointer indirection.
 *
 * Layout in GPU memory:
 *   cntr_wb[0], cntr_wb[1], ..., cntr_wb[counter_count-1]    (contiguous)
 *   tgt_ct_wb[0], tgt_ct_wb[1], ..., tgt_ct_wb[tgt_ct_count-1] (contiguous)
 *   signals[0], signals[1], ..., signals[signal_count-1]     (contiguous)
 */
struct fi_cxi_ki_writebacks {
    /* Local counters - contiguous array (KI requires batch allocation) */
    volatile struct c_ct_writeback *cntr_wb;  /* [counter_count] Direct array */
    void              *cntr_mmio_base;        /* Base doorbell address */
    size_t             cntr_mmio_stride;      /* Stride between doorbells */
    uint16_t          *cntr_ct_idx;           /* [counter_count] CT indices for commands */
    uint32_t           counter_count;

    /* Target CTs - contiguous array (KI requires batch allocation) */
    volatile struct c_ct_writeback *tgt_ct_wb; /* [tgt_ct_count] Direct array */
    uint32_t            tgt_ct_count;

    /* Signals - contiguous array (SW-managed, peers FAA here) */
    volatile uint64_t  *signals;      /* [signal_count] Direct array in GPU memory */
    uint32_t            signal_count;
};

/* Static configuration */
struct fi_cxi_ki_config {
    uint64_t local_iova;        /* Local MR base IOVA */
    uint32_t local_lac;         /* Local LAC */
    uint32_t local_nid;
    uint32_t local_pid;
    uint8_t  local_pid_bits;
    uint16_t local_vni;
};

/* Complete metadata structure */
struct fi_cxi_ki_meta {
    /* HOT: Command queues */
    struct fi_cxi_ki_cmdq_config cmdq_config[FI_KI_MAX_CONTEXTS];
    struct fi_cxi_ki_cmdq_hot    cmdq_hot[FI_KI_MAX_CONTEXTS];
    uint32_t                     context_count;

    /* HOT: Peer data (SoA) */
    struct fi_cxi_ki_peer_soa    peers;

    /* WARM: Completion writebacks */
    struct fi_cxi_ki_writebacks  wb;

    /* COLD: Configuration */
    struct fi_cxi_ki_config      config;
} __attribute__((aligned(FI_KI_CACHE_LINE)));

typedef struct fi_cxi_ki_meta *fi_ki_meta_t;
```

### Peer Data Access Macros

```c
/* Routing - always needed */
#define FI_KI_PEER_DFA(meta, idx)         ((meta)->peers.dfa[idx])
#define FI_KI_PEER_DFA_EXT(meta, idx)     ((meta)->peers.dfa_ext[idx])
#define FI_KI_PEER_INDEX_EXT(meta, idx)   ((meta)->peers.index_ext[idx])

/* Data memory region */
#define FI_KI_PEER_MR_BASE(meta, idx)     ((meta)->peers.mr_base[idx])
#define FI_KI_PEER_MR_KEY(meta, idx)      ((meta)->peers.mr_key[idx])

/* Signal memory region (Model B) */
#define FI_KI_PEER_SIGNAL_BASE(meta, idx) ((meta)->peers.signal_base[idx])
#define FI_KI_PEER_SIGNAL_KEY(meta, idx)  ((meta)->peers.signal_key[idx])
```

---

## Usage Flow

### 1. Setup Phase (Host)

```c
int setup_ki_endpoint(struct fid_ep *ep, int gpu_dev_id,
                      uint32_t num_counters, uint32_t num_target_cts) {
    struct fi_cxi_ki_ops *ki_ops;
    int ret;

    /* Get KI operations */
    ret = fi_open_ops(&ep->fid, FI_CXI_KI_OPS_1, 0, (void **)&ki_ops, NULL);
    if (ret) return ret;

    /* Get endpoint info */
    struct fi_cxi_ki_ep_info ep_info;
    ret = ki_ops->get_ep_info(&ep->fid, &ep_info);
    if (ret) return ret;

    /* Get command queue info */
    struct fi_cxi_ki_cmdq_info cmdq_info;
    ret = ki_ops->get_cmdq_info(&ep->fid, 0, &cmdq_info);
    if (ret) return ret;

    /* Resolve targets */
    struct fi_cxi_ki_target_info *targets;
    targets = calloc(peer_count, sizeof(*targets));
    for (int i = 0; i < peer_count; i++) {
        ret = ki_ops->resolve_target(&ep->fid, fi_addrs[i], &targets[i]);
        if (ret) return ret;
    }

    /* Allocate CONTIGUOUS GPU memory for counter writebacks */
    void *cntr_gpu_wb;
    cudaMalloc(&cntr_gpu_wb, num_counters * sizeof(struct c_ct_writeback));

    /* Batch-allocate counters with contiguous writeback buffers */
    struct fid_cntr **cntrs = calloc(num_counters, sizeof(*cntrs));
    struct fi_cxi_ki_cntr_batch_info cntr_info;
    ret = ki_ops->alloc_counters_batch(&ep->fid, num_counters,
                                        cntr_gpu_wb, cntrs, &cntr_info);
    if (ret) return ret;

    /* Allocate CONTIGUOUS GPU memory for target CT writebacks (Model A) */
    void *tgt_ct_gpu_wb;
    cudaMalloc(&tgt_ct_gpu_wb, num_target_cts * sizeof(struct c_ct_writeback));

    /* Batch-allocate target CTs */
    struct fi_cxi_ki_target_ct_info tgt_ct_info;
    ret = ki_ops->alloc_target_cts_batch(&ep->fid, num_target_cts,
                                          tgt_ct_gpu_wb, &tgt_ct_info);
    if (ret) return ret;

    /* Assemble metadata - now with direct array pointers */
    struct fi_cxi_ki_meta *meta;
    /* ... see Metadata Assembly below ... */

    return 0;
}
```

### 2. Metadata Assembly

```c
int assemble_ki_metadata(
    struct fi_cxi_ki_cmdq_info *cmdq_info,
    struct fi_cxi_ki_ep_info *ep_info,
    struct fi_cxi_ki_target_info *targets,
    struct fi_cxi_ki_mr_info *peer_mrs,
    uint64_t *peer_signal_bases,
    uint64_t *peer_signal_keys,
    uint32_t peer_count,
    struct ct_info *cntrs, uint32_t cntr_count,
    struct ct_info *tgt_cts, uint32_t tgt_ct_count,
    uint64_t *signals_gpu, uint32_t signal_count,
    fi_ki_meta_t *meta_out
) {
    /* Calculate size for SoA peer arrays */
    size_t peer_arrays_size = peer_count * (
        sizeof(uint64_t) +   /* dfa */
        sizeof(uint8_t) +    /* dfa_ext (padded for alignment) */
        sizeof(uint32_t) +   /* index_ext */
        sizeof(uint64_t) +   /* mr_base */
        sizeof(uint64_t) +   /* mr_key */
        sizeof(uint64_t) +   /* signal_base */
        sizeof(uint64_t)     /* signal_key */
    );

    /* Calculate size for writeback pointer arrays */
    size_t wb_arrays_size =
        cntr_count * sizeof(uint64_t*) * 2 +     /* cntr_success, cntr_failure */
        cntr_count * sizeof(void*) +             /* cntr_mmio */
        cntr_count * sizeof(uint16_t) +          /* cntr_ct_idx */
        tgt_ct_count * sizeof(uint64_t*) * 2;    /* tgt_ct_success, tgt_ct_failure */

    size_t total = sizeof(struct fi_cxi_ki_meta) + peer_arrays_size + wb_arrays_size;

    /* Allocate GPU memory for metadata + trailing arrays */
    struct fi_cxi_ki_meta *meta;
    cudaMalloc(&meta, total);

    /* Allocate host staging buffer */
    struct fi_cxi_ki_meta *host_meta = calloc(1, total);
    char *array_ptr = (char*)host_meta + sizeof(struct fi_cxi_ki_meta);

    /* Populate command queue info */
    host_meta->context_count = 1;
    host_meta->cmdq_config[0].buf = cmdq_info->cmdq_buf;
    host_meta->cmdq_config[0].mask = cmdq_info->cmdq_mask;
    host_meta->cmdq_config[0].slot_size = cmdq_info->cmd_slot_size;
    host_meta->cmdq_hot[0].wp_addr = cmdq_info->wp_addr;
    host_meta->cmdq_hot[0].wp = 0;

    /* Setup peer SoA - pointers point into GPU memory */
    char *gpu_array_ptr = (char*)meta + sizeof(struct fi_cxi_ki_meta);
    host_meta->peers.count = peer_count;
    host_meta->peers.dfa = (uint64_t*)gpu_array_ptr;
    /* ... similar for other peer arrays ... */

    /* Copy peer data into trailing arrays */
    uint64_t *host_dfa = (uint64_t*)array_ptr;
    for (uint32_t i = 0; i < peer_count; i++) {
        host_dfa[i] = targets[i].dfa;
    }
    /* ... similar for dfa_ext, index_ext, mr_base, mr_key, signal_base, signal_key ... */

    /* Setup writeback arrays - CONTIGUOUS thanks to batch allocation */
    host_meta->wb.cntr_wb = (struct c_ct_writeback*)cntr_gpu_wb;
    host_meta->wb.cntr_mmio_base = cntr_info.mmio_base;
    host_meta->wb.cntr_mmio_stride = cntr_info.mmio_stride;
    host_meta->wb.cntr_ct_idx = cntr_info.ct_indices;  /* Copy to GPU too */
    host_meta->wb.counter_count = cntr_count;

    /* Target CTs - also contiguous */
    host_meta->wb.tgt_ct_wb = (struct c_ct_writeback*)tgt_ct_gpu_wb;
    host_meta->wb.tgt_ct_count = tgt_ct_count;

    /* Signals - contiguous array for remote FAA */
    host_meta->wb.signals = signals_gpu;
    host_meta->wb.signal_count = signal_count;

    /* Populate config */
    host_meta->config.local_nid = ep_info->nid;
    host_meta->config.local_pid = ep_info->pid;
    host_meta->config.local_pid_bits = ep_info->pid_bits;
    host_meta->config.local_vni = ep_info->vni;

    /* Copy to GPU */
    cudaMemcpy(meta, host_meta, total, cudaMemcpyHostToDevice);
    free(host_meta);

    *meta_out = meta;
    return 0;
}
```

**Memory Layout Rationale:**

The metadata references several GPU memory regions:

1. **Metadata structure** (`fi_cxi_ki_meta`) - Contains pointers to other regions
2. **Peer SoA arrays** - Contiguous arrays for coalesced warp access
3. **Counter writebacks** - Contiguous array (requires `alloc_counters_batch`)
4. **Target CT writebacks** - Contiguous array (requires `alloc_target_cts_batch`)
5. **Signal array** - Contiguous array for remote FAA

**Why batch allocation is required:**

Without batch allocation, each `fi_cntr_open()` call allocates an independent
writeback buffer at a provider-chosen address. This results in non-contiguous
memory that requires pointer indirection to access.

With `alloc_counters_batch()`, the application provides a pre-allocated contiguous
GPU buffer. The provider allocates CTs with writebacks placed at known offsets
within this buffer, enabling direct `cntr_wb[idx]` array access.

Peer data uses SoA layout so that when 32 warp threads each access a different
peer index, memory loads are coalesced. See [GPU API - Memory Access Patterns](fi_cxi_ki_gpu.md#memory-access-patterns).

### 3. Post-Kernel Sync (Host)

```c
int sync_after_kernel(struct fid_ep *ep, struct fi_cxi_ki_ops *ki_ops,
                      int ctx_idx, uint32_t cmds_posted) {
    /* Inform provider of commands posted by GPU kernel */
    return ki_ops->sync_cmdq_wp(&ep->fid, ctx_idx, cmds_posted);
}
```

### 4. Resource Lifecycle

KI resources must be managed carefully to avoid use-after-free and resource leaks.

#### Setup Order

```
1. fi_fabric(), fi_domain()           - Standard libfabric setup
2. fi_cntr_open() for KI counters     - Allocate counters
3. fi_open_ops(FI_CXI_COUNTER_OPS)    - Get counter extension
4. cntr_ops->set_wb_buffer(gpu_mem)   - Point writeback to GPU memory
5. fi_endpoint(), fi_ep_bind()        - Create and bind endpoint
6. fi_open_ops(FI_CXI_KI_OPS_1)       - Get KI extension
7. ki_ops->get_*_info()               - Gather all metadata
8. Assemble fi_cxi_ki_meta            - Build metadata structure
9. cudaMemcpy() to GPU                - Copy metadata to device
```

#### Teardown Order

```
1. Wait for all GPU kernels to complete
2. ki_ops->sync_cmdq_wp()             - Sync final write pointer
3. Wait for all in-flight operations   - fi_cntr_wait() until counters match
4. cudaFree(gpu_meta)                 - Free GPU metadata
5. fi_close(&cntr->fid)               - Close counters
6. fi_close(&ep->fid)                 - Close endpoint
7. fi_close(&domain->fid)             - Close domain
```

#### Critical Rules

1. **Never free GPU writeback buffers while operations are in-flight**
   - The NIC DMAs to these buffers asynchronously
   - Wait for all counters to reach expected values first

2. **Sync write pointer before teardown**
   - Host must know how many commands GPU posted
   - Call `sync_cmdq_wp()` after kernel completes

3. **GPU kernel abort handling**
   - If a kernel is killed (OOM, timeout, etc.), host cannot know write pointer
   - Reset the endpoint and re-setup resources

4. **Multi-GPU considerations**
   - Each GPU needs its own metadata copy
   - Counters can be shared if writeback buffers are accessible to all GPUs

#### Example: Safe Teardown

```c
int teardown_ki_resources(struct ki_resources *res) {
    int ret;

    /* 1. Ensure GPU kernel has completed */
    cudaDeviceSynchronize();

    /* 2. Sync write pointer */
    ret = res->ki_ops->sync_cmdq_wp(&res->ep->fid, 0, res->cmds_posted);
    if (ret) return ret;

    /* 3. Wait for all operations to complete */
    uint64_t expected = res->cmds_posted;
    ret = fi_cntr_wait(res->cntr, expected, -1);
    if (ret) return ret;

    /* 4. Free GPU resources */
    cudaFree(res->gpu_meta);
    cudaFree(res->gpu_wb_buffer);

    /* 5. Close libfabric resources (reverse order of creation) */
    fi_close(&res->cntr->fid);
    fi_close(&res->ep->fid);
    fi_close(&res->domain->fid);
    fi_close(&res->fabric->fid);

    return 0;
}
```

---

## Provider Implementation

### Internal Structures

The provider uses existing structures for KI. The key libcxi structures (`struct cxi_cq`, `struct cxi_ct`) are public and can be accessed directly by the provider.

```c
/* Extends cxip_cntr for KI - wraps libcxi's cxil_ct */
struct cxip_cntr {
    struct fid_cntr cntr_fid;
    struct cxip_domain *domain;
    struct cxi_ct *ct;               /* libcxi CT handle */
    struct c_ct_writeback *wb;       /* 8-byte writeback buffer */
    uint64_t wb_device;              /* Device memory handle (for GPU) */
    enum fi_hmem_iface wb_iface;     /* Memory interface type */
    /* ... */
};

/* KI endpoint extension */
struct cxip_ki_ep {
    struct cxip_ep *base_ep;

    /* GPU device */
    int gpu_dev_id;

    /* Counters with GPU writeback */
    struct cxip_cntr **ki_cntrs;
    uint32_t ki_cntr_count;

    /* Metadata */
    struct fi_cxi_ki_meta *meta;
    uint64_t meta_gpu_va;
};
```

### Operations Implementation

```c
static int cxip_ki_get_cmdq_info(struct fid *fid, int tx_ctx_index,
                                  struct fi_cxi_ki_cmdq_info *info)
{
    struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
    struct cxip_ep_obj *ep_obj = ep->ep_obj;
    struct cxip_cmdq *txq = ep_obj->txq;

    /* Access public struct cxi_cq fields directly */
    struct cxi_cq *cq = txq->dev_cmdq;

    info->cmdq_buf = cq->cmds32;         /* mmap'd command buffer (32-byte slots) */
    info->cmdq_size = cq->size32;        /* number of 32-byte slots */
    info->cmdq_mask = cq->size32 - 1;
    info->cmd_slot_size = 32;            /* CXI base command slot size */
    info->wp_addr = cq->wp_addr;         /* mmap'd write pointer CSR */

    return FI_SUCCESS;
}

static int cxip_ki_get_ep_info(struct fid *fid, struct fi_cxi_ki_ep_info *info)
{
    struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
    struct cxip_ep_obj *ep_obj = ep->ep_obj;

    info->nid = ep_obj->src_addr.nic;
    info->pid = ep_obj->src_addr.pid;
    info->pid_bits = ep_obj->domain->iface->dev->info.pid_bits;
    info->vni = ep_obj->src_addr.vni;

    return FI_SUCCESS;
}

static int cxip_ki_resolve_target(struct fid *fid, fi_addr_t addr,
                                   struct fi_cxi_ki_target_info *info)
{
    struct cxip_ep *ep = container_of(fid, struct cxip_ep, ep.fid);
    struct cxip_ep_obj *ep_obj = ep->ep_obj;
    struct cxip_addr peer_addr;
    int ret;

    /* Lookup peer address from AV */
    ret = cxip_av_lookup_addr(ep_obj->av, addr, &peer_addr);
    if (ret)
        return ret;

    /* Build DFA for peer */
    cxi_build_dfa(peer_addr.nic, peer_addr.pid,
                  ep_obj->domain->iface->dev->info.pid_bits,
                  0, /* pid_offset */
                  &info->dfa, &info->index_ext);
    info->dfa_ext = 0;

    return FI_SUCCESS;
}

static int cxip_ki_get_mr_info(struct fid *fid, struct fi_cxi_ki_mr_info *info)
{
    struct cxip_mr *mr = container_of(fid, struct cxip_mr, mr_fid.fid);

    /*
     * Access MR info from libcxi's cxi_md structure.
     * See fi_cxi_ki_driver.md#7-memory-descriptor-md for structure details.
     */
    info->iova = mr->md->md->iova;    /* NIC-visible IO virtual address */
    info->lac = mr->md->md->lac;      /* Logical access context */
    info->key = mr->key;              /* MR key for remote access */

    return FI_SUCCESS;
}

static struct fi_cxi_ki_ops cxip_ki_ops = {
    .get_cmdq_info = cxip_ki_get_cmdq_info,
    .get_ep_info = cxip_ki_get_ep_info,
    .resolve_target = cxip_ki_resolve_target,
    .get_mr_info = cxip_ki_get_mr_info,
    .sync_cmdq_wp = cxip_ki_sync_cmdq_wp,
};

int cxip_ep_ops_open(struct fid *fid, const char *name, uint64_t flags,
                     void **ops, void *context)
{
    if (!strcmp(name, FI_CXI_KI_OPS_1)) {
        *ops = &cxip_ki_ops;
        return FI_SUCCESS;
    }
    /* ... existing operations ... */
    return -FI_ENOSYS;
}
```

---

## Integration with Existing Counter Extension

The KI API leverages the existing `fi_cxi_cntr_ops` for counter GPU access:

```c
/* Model A: GPU polls counter writeback for receive completion */
int setup_model_a_counter(struct fid_domain *domain, void *gpu_wb_buf) {
    struct fid_cntr *cntr;
    struct fi_cxi_cntr_ops *cntr_ops;

    /* Create counter */
    fi_cntr_open(domain, &cntr_attr, &cntr, NULL);

    /* Get counter operations */
    fi_open_ops(&cntr->fid, FI_CXI_COUNTER_OPS, 0, (void **)&cntr_ops, NULL);

    /* Set GPU writeback buffer */
    cntr_ops->set_wb_buffer(&cntr->fid, gpu_wb_buf,
                             sizeof(struct c_ct_writeback));

    /* GPU kernel can now poll gpu_wb_buf using fi_cxi_cntr_wb_read() */
    return 0;
}

/* Model B: GPU writes counter doorbell to trigger operations */
int setup_model_b_counter(struct fid_domain *domain, void **mmio_out) {
    struct fid_cntr *cntr;
    struct fi_cxi_cntr_ops *cntr_ops;
    void *mmio_addr;
    size_t mmio_len;

    /* Create counter */
    fi_cntr_open(domain, &cntr_attr, &cntr, NULL);

    /* Get counter operations */
    fi_open_ops(&cntr->fid, FI_CXI_COUNTER_OPS, 0, (void **)&cntr_ops, NULL);

    /* Get MMIO doorbell address */
    cntr_ops->get_mmio_addr(&cntr->fid, &mmio_addr, &mmio_len);

    /* Map to GPU and return */
    /* GPU kernel can write using fi_cxi_cntr_add() equivalent */
    *mmio_out = mmio_addr;
    return 0;
}
```

---

## API Summary

### Existing APIs (from fi_cxi_ext.h)

| API | Purpose |
|-----|---------|
| `fi_cxi_cntr_ops.set_wb_buffer` | Set counter writeback to GPU memory |
| `fi_cxi_cntr_ops.get_mmio_addr` | Get counter doorbell for GPU writes |
| `fi_cxi_cntr_wb_read()` | Read success counter from writeback buffer |
| `fi_cxi_cntr_wb_readerr()` | Read failure counter from writeback buffer |
| `fi_cxi_cntr_add()` | Increment success counter via MMIO doorbell |
| `fi_cxi_cntr_adderr()` | Increment failure counter via MMIO doorbell |
| `fi_cxi_cntr_set()` | Reset success counter via MMIO (value must be 0) |
| `fi_cxi_cntr_seterr()` | Reset failure counter via MMIO (value must be 0) |
| `fi_cxi_gen_cntr_success()` | Generate pollable counter value |
| `fi_cxi_get_cntr_add_addr()` | Get MMIO address for counter increment |
| `fi_cxi_get_cntr_reset_addr()` | Get MMIO address for counter reset |
| `fi_cxi_dom_ops.get_dwq_depth` | Query deferred work queue capacity |
| `FI_CXI_CNTR_SUCCESS_MAX` | Maximum success counter value (2^48-1) |
| `FI_CXI_CNTR_FAILURE_MAX` | Maximum failure counter value (2^7-1) |

### New KI APIs

| API | Purpose |
|-----|---------|
| `fi_cxi_ki_ops.get_cmdq_info` | Get command queue addresses for GPU |
| `fi_cxi_ki_ops.get_ep_info` | Get endpoint NIC/PID/VNI info |
| `fi_cxi_ki_ops.resolve_target` | Resolve fi_addr_t to DFA routing |
| `fi_cxi_ki_ops.get_mr_info` | Get MR IOVA/LAC/key |
| `fi_cxi_ki_ops.sync_cmdq_wp` | Sync write pointer after GPU kernel |
| `fi_cxi_ki_ops.alloc_counters_batch` | Batch-allocate counters with contiguous GPU writeback |
| `fi_cxi_ki_ops.alloc_target_cts_batch` | Batch-allocate target CTs with contiguous GPU writeback |
