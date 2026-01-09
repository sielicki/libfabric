# KI API v4: libcxi Reference

This document is **reference documentation** for existing libcxi APIs and structures used by KI. **No libcxi changes are required for KI** — the relevant structures are already public.

For the libfabric provider that uses these APIs, see [Libfabric API](fi_cxi_ki_libfabric.md). For kernel driver implementation, see [CXI Driver](fi_cxi_ki_driver.md).

---

## Summary: No Changes Needed

KI uses existing public structures from `cxi_prov_hw.h`:

| Structure | KI-Relevant Fields | Access |
|-----------|-------------------|--------|
| `struct cxi_cq` | `cmds`, `size`, `wp_addr` | Direct field access |
| `struct cxi_ct` | `doorbell`, `ctn`, `wb` | Direct field access |

The libfabric provider already accesses these structures directly (e.g., `cntr->ct->doorbell` in the existing `fi_cxi_cntr_ops.get_mmio_addr` implementation).

---

## Existing API (Reference)

The following existing APIs are used for KI resource setup.

### Memory Registration with GPU/dmabuf Support

GPU memory registration is already supported via `cxil_map()` using dmabuf hints:

```c
/* From libcxi.h */
int cxil_map(struct cxil_lni *lni, void *va, size_t len,
             uint32_t flags, struct cxi_md_hints *hints,
             struct cxi_md **md);

/* From uapi/misc/cxi.h - dmabuf hints structure */
struct cxi_md_hints {
    int dmabuf_fd;           /* dmabuf file descriptor */
    size_t dmabuf_offset;    /* offset within dmabuf */
    bool dmabuf_valid;       /* true if dmabuf_fd is valid */
    unsigned int huge_shift; /* hugepage shift hint */
    unsigned int page_shift; /* page shift hint */
};
```

**Usage for GPU memory:**
```c
/* CUDA/HIP/oneAPI export dmabuf fd */
int dmabuf_fd = /* exported from GPU runtime */;
void *gpu_va = /* GPU virtual address */;

struct cxi_md_hints hints = {
    .dmabuf_fd = dmabuf_fd,
    .dmabuf_offset = (uintptr_t)gpu_va - (uintptr_t)base,
    .dmabuf_valid = true,
};

struct cxi_md *md;
int ret = cxil_map(lni, gpu_va, length, CXI_MAP_PIN, &hints, &md);
```

### Counting Events with Doorbell

CTs are allocated with a writeback buffer and expose a memory-mapped doorbell:

```c
/* From libcxi.h */
int cxil_alloc_ct(struct cxil_lni *lni, struct c_ct_writeback *wb,
                  struct cxi_ct **ct);

int cxil_ct_wb_update(struct cxi_ct *ct, struct c_ct_writeback *wb);

int cxil_destroy_ct(struct cxi_ct *ct);
```

**Internal structure (from libcxi_priv.h):**
```c
struct cxil_ct {
    struct cxil_lni_priv *lni_priv;
    unsigned int ct_hndl;
    void *doorbell;          /* mmap'd doorbell register */
    size_t doorbell_len;
    struct cxi_ct ct;        /* hardware CT structure */
};

/* Hardware CT structure (from cxi_prov_hw.h) */
struct cxi_ct {
    uint64_t *doorbell;      /* doorbell pointer for direct writes */
    unsigned int ctn;        /* CT number */
    struct c_ct_writeback *wb; /* writeback buffer */
};
```

**Doorbell operations (inline functions from cxi_prov_hw.h):**
```c
void cxi_ct_inc_success(struct cxi_ct *ct, uint64_t count);
void cxi_ct_inc_failure(struct cxi_ct *ct, uint64_t count);
void cxi_ct_reset_success(struct cxi_ct *ct);
void cxi_ct_reset_failure(struct cxi_ct *ct);
```

For driver-level CT structure details (including `c_ct_writeback`), see [CXI Driver - CT Allocation](fi_cxi_ki_driver.md#2-counting-event-ct-allocation).

### Triggered Command Queues

Triggered operations are supported via communication profiles and CQ flags:

```c
/* Triggered CP allocation */
int cxil_alloc_trig_cp(struct cxil_lni *lni, unsigned int vni,
                       enum cxi_traffic_class tc,
                       enum cxi_traffic_class_type tc_type,
                       enum cxi_trig_cp cp_type,
                       struct cxi_cp **cp);

/* CP types for triggered operations */
enum cxi_trig_cp {
    TRIG_LCID,      /* Triggered LCID only */
    NON_TRIG_LCID,  /* Non-triggered LCID only */
    ANY_LCID,       /* Either type */
};
```

**CQ allocation with triggered support:**
```c
struct cxi_cq_alloc_opts opts = {
    .count = 1024,
    .flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS,
};

struct cxi_cq *cmdq;
int ret = cxil_alloc_cmdq(lni, evtq, &opts, &cmdq);
```

**Internal CQ structure (from libcxi_priv.h):**
```c
struct cxil_cq {
    struct cxil_lni_priv *lni_priv;
    unsigned int cq_hndl;
    void *cmds;              /* mmap'd command buffer */
    size_t cmds_len;
    struct cxi_cq hw;        /* hardware CQ structure */
};

/* Hardware CQ structure (from cxi_prov_hw.h) - simplified */
struct cxi_cq {
    unsigned int size;           /* Total queue size */
    uint64_t *wp_addr;           /* Memory mapped write pointer CSR */
    uint8_t *ll_64;              /* Low-latency 64-byte write region */
    uint8_t *ll_128a;            /* Low-latency 128-byte aligned region */
    uint8_t *ll_128u;            /* Low-latency 128-byte unaligned region */
    union {
        struct cxi_cmd32 *cmds32;    /* 32-byte command slots */
        volatile struct c_cq_status *status;  /* CQ status (first 8 slots) */
    };
    uint64_t rp32;               /* 32-byte read pointer */
    unsigned int idx;            /* CQ index (TX: 0-1023, TGT: 1024-1535) */
    unsigned int size32;         /* Size in 32-byte slots */
    uint64_t wp32;               /* 32-byte write pointer (software) */
    uint64_t hw_wp32;            /* 32-byte write pointer (hardware) */
};
```

**Command Slot Sizes:**
- Base slot: 32 bytes (`struct cxi_cmd32`)
- Standard DMA: 64 bytes (2 slots)
- Triggered DMA: 128 bytes (4 slots)

For driver-level CQ structure details, see [CXI Driver - CQ Allocation](fi_cxi_ki_driver.md#4-command-queue-cq-allocation).

---

## GPU Writeback Configuration

To enable GPU polling of CT values, the writeback buffer must be in GPU-accessible memory. This is achieved using **existing APIs**:

```c
/* Option 1: Allocate CT, then update writeback to GPU memory */
struct c_ct_writeback *gpu_wb = /* GPU memory, exported via dmabuf */;
struct cxi_ct *ct;
cxil_alloc_ct(lni, gpu_wb, &ct);  /* Pass GPU VA directly */

/* Option 2: Update existing CT's writeback buffer */
cxil_ct_wb_update(ct, gpu_wb);    /* Change writeback to GPU memory */
```

The driver's `CXI_OP_CT_WB_UPDATE` ioctl handles the synchronization required to safely update the writeback address.

---

## PTE Configuration for Receive Counting

For Model A (receiver signals), PtlTEs must be configured to count incoming operations.

### Target CT Routing Modes

| Mode | Match Bits Usage | CT Allocation | Use Case |
|------|-----------------|---------------|----------|
| **Aggregate** | Ignored | 1 CT total | Simple count of all incoming PUTs |
| **Per-Peer** | Sender's rank | 1 CT per peer | Track arrivals from each sender |
| **Custom** | Application-defined | Application-defined | Complex routing patterns |

### Mode 1: Aggregate CT (Simplest)

All incoming PUTs increment a single CT. Use with `fi_cxi_inject_put_simple()` on sender.

```c
/* Receiver setup: single CT for all incoming PUTs */
int setup_aggregate_ct_receiver(struct cxil_lni *lni, int gpu_dev_id) {
    struct cxi_ct *ct;

    /* 1. Allocate single CT with GPU writeback */
    void *gpu_wb = gpu_alloc(sizeof(struct c_ct_writeback));
    cxil_alloc_ct(lni, gpu_wb, &ct);

    /* 2. Allocate non-matching PTE (no match bit routing) */
    struct cxi_pt_alloc_opts opts = { .is_matching = false };
    struct cxil_pte *pte;
    cxil_alloc_pte(lni, evtq, &opts, &pte);

    /* 3. Append LE with CT binding */
    union c_cmdu cmd = {};
    cmd.append.command.opcode = C_CMD_TGT_APPEND;
    cmd.append.ptlte_index = pte->ptn;
    cmd.append.ct = ct->ctn;
    cmd.append.ct_success = true;
    cmd.append.start = buffer_base;
    cmd.append.length = buffer_size;
    /* ... */
    cxi_cq_emit_target(target_cmdq, &cmd);

    return 0;
}
```

### Mode 2: Per-Peer CT Routing

Each sender's PUTs increment a different CT based on match bits. Use with `fi_cxi_inject_put_tagged()` on sender, passing sender's rank as `match_bits`.

```c
/* Receiver setup: one CT per peer for per-sender tracking */
int setup_per_peer_ct_receiver(struct cxil_lni *lni, int gpu_dev_id,
                                int num_peers) {
    struct cxi_ct *cts[MAX_PEERS];

    /* 1. Allocate one CT per peer, each with GPU writeback */
    for (int i = 0; i < num_peers; i++) {
        void *gpu_wb = gpu_alloc(sizeof(struct c_ct_writeback));
        cxil_alloc_ct(lni, gpu_wb, &cts[i]);
    }

    /* 2. Allocate MATCHING PTE for match bit routing */
    struct cxi_pt_alloc_opts opts = {
        .is_matching = true,
        .en_event_match = true,  /* Route by match bits */
    };
    struct cxil_pte *pte;
    cxil_alloc_pte(lni, evtq, &opts, &pte);

    /* 3. Append one LE per peer with match bits = peer rank */
    for (int i = 0; i < num_peers; i++) {
        union c_cmdu cmd = {};
        cmd.append.command.opcode = C_CMD_TGT_APPEND;
        cmd.append.ptlte_index = pte->ptn;
        cmd.append.ct = cts[i]->ctn;     /* CT for this peer */
        cmd.append.ct_success = true;
        cmd.append.match_bits = i;        /* Match on sender's rank */
        cmd.append.ignore_bits = 0;       /* Exact match */
        cmd.append.start = buffer_base;
        cmd.append.length = buffer_size;
        /* ... */
        cxi_cq_emit_target(target_cmdq, &cmd);
    }

    return 0;
}
```

**Sender side (using tagged PUT):**
```c
/* GPU kernel on sender: use my_rank as match_bits */
uint32_t my_rank = get_my_rank();
fi_cxi_inject_put_tagged(meta, ctx, peer_idx, src, dst_off, len,
                          my_rank,  /* match_bits = sender's rank */
                          FI_KI_COOP_THREAD, counter_idx);
```

### PTE and LE Structures

```c
/* From libcxi.h */
int cxil_alloc_pte(struct cxil_lni *lni, struct cxi_eq *evtq,
                   struct cxi_pt_alloc_opts *opts,
                   struct cxil_pte **pte);

int cxil_map_pte(struct cxil_pte *pte, struct cxil_domain *domain,
                 unsigned int pid_offset, bool is_multicast,
                 struct cxil_pte_map **pte_map);
```

**PTE allocation options (from uapi/misc/cxi.h):**
```c
struct cxi_pt_alloc_opts {
    uint64_t en_event_match     : 1;   /* Enable match bit routing */
    uint64_t is_matching        : 1;   /* Matching vs non-matching portal */
    uint64_t use_logical        : 1;   /* Logical addressing */
    /* ... other options ... */
};
```

**LE append command fields for CT binding:**

```c
/* Append LE with CT binding (emitted via cxi_cq_emit_target) */
union c_cmdu cmd = {};
cmd.append.command.opcode = C_CMD_TGT_APPEND;
cmd.append.ptlte_index = pte->ptn;  /* Portal table entry index */
cmd.append.ct = ct->ctn;             /* CT to increment on receive */
cmd.append.ct_success = true;        /* Increment CT on success */
cmd.append.match_bits = ...;         /* Match bits for routing */
cmd.append.ignore_bits = ...;        /* Bits to ignore in matching */
cmd.append.start = buffer_base;      /* Receive buffer base */
cmd.append.length = buffer_size;     /* Receive buffer length */

cxi_cq_emit_target(target_cmdq, &cmd);
```

---

## Usage Flow

### Model A: Receiver Signals (GPU polls CT writeback)

```c
int setup_model_a_receiver(struct cxil_lni *lni, int gpu_dev_id) {
    struct cxi_ct *ct;
    struct cxil_pte *pte;
    int ret;

    /* 1. Allocate GPU memory for CT writeback */
    void *wb_gpu;
    int wb_dmabuf_fd;
    /* ... allocate via CUDA/HIP and export dmabuf ... */

    /* 2. Allocate CT with GPU-accessible writeback */
    ret = cxil_alloc_ct_gpu(lni, wb_gpu, wb_dmabuf_fd, &ct);
    if (ret) return ret;

    /* 3. Allocate PTE */
    struct cxi_pt_alloc_opts pt_opts = { .is_matching = false };
    ret = cxil_alloc_pte(lni, evtq, &pt_opts, &pte);
    if (ret) goto err_free_ct;

    /* 4. Map PTE to domain */
    ret = cxil_map_pte(pte, domain, pid_offset, false, &pte_map);
    if (ret) goto err_free_pte;

    /* 5. Enable PTE and bind CT via append commands */
    /* ... emit TGQ commands ... */

    /* 6. Pass CT writeback address to GPU kernel */
    /* GPU kernel polls wb_gpu->success for completion */

    return 0;

err_free_pte:
    cxil_destroy_pte(pte);
err_free_ct:
    cxil_destroy_ct(ct);
    return ret;
}
```

### Model B: Initiator Signals (GPU writes doorbell)

```c
int setup_model_b_initiator(struct cxil_lni *lni, int gpu_dev_id) {
    struct cxi_cq *cmdq;
    struct cxi_ct *ct;
    int ret;

    /* 1. Allocate CQ with triggered command support */
    struct cxi_cq_alloc_opts cq_opts = {
        .count = 1024,
        .flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS,
    };
    ret = cxil_alloc_cmdq(lni, evtq, &cq_opts, &cmdq);
    if (ret) return ret;

    /* 2. Allocate CT for GPU signaling */
    struct c_ct_writeback wb = {};
    ret = cxil_alloc_ct(lni, &wb, &ct);
    if (ret) goto err_free_cq;

    /* 3. Get doorbell info */
    void *doorbell_cpu_va;
    size_t doorbell_len;
    ret = cxil_get_ct_doorbell_info(ct, &doorbell_cpu_va, &doorbell_len);
    if (ret) goto err_free_ct;

    /* 4. Map doorbell to GPU address space */
    /* ... use CUDA/HIP cudaHostRegister or similar ... */

    /* 5. Pre-program triggered operations */
    /* ... emit triggered PUT/ATOMIC commands with CT threshold ... */

    /* 6. Pass doorbell GPU VA to GPU kernel */
    /* GPU kernel writes to doorbell to trigger NIC operations */

    return 0;

err_free_ct:
    cxil_destroy_ct(ct);
err_free_cq:
    cxil_destroy_cmdq(cmdq);
    return ret;
}
```

---

## API Summary

### KI-Relevant Existing APIs

| Function | Purpose |
|----------|---------|
| `cxil_map` | Map memory (CPU or GPU via dmabuf) for NIC access |
| `cxil_unmap` | Unmap memory descriptor |
| `cxil_alloc_ct` | Allocate CT with writeback buffer (supports GPU memory) |
| `cxil_ct_wb_update` | Update CT writeback buffer pointer (supports GPU memory) |
| `cxil_destroy_ct` | Free CT |
| `cxil_alloc_cmdq` | Allocate command queue |
| `cxil_destroy_cmdq` | Free command queue |
| `cxil_alloc_trig_cp` | Allocate triggered communication profile |
| `cxil_alloc_pte` | Allocate portal table entry |
| `cxil_map_pte` | Map PTE to domain |
| `cxil_destroy_pte` | Free PTE |

### KI-Relevant Public Structures

| Structure | Fields Used by KI |
|-----------|-------------------|
| `struct cxi_cq` | `cmds32`, `size32`, `wp_addr`, `wp32` |
| `struct cxi_ct` | `doorbell`, `ctn`, `wb` |

---

## Hardware Access Functions

The following inline functions from `cxi_prov_hw.h` are used for direct hardware access:

```c
/* CT operations */
void cxi_ct_init(struct cxi_ct *ct, struct c_ct_writeback *wb,
                 unsigned int ctn, void *doorbell);
void cxi_ct_inc_success(struct cxi_ct *ct, uint64_t count);
void cxi_ct_inc_failure(struct cxi_ct *ct, uint64_t count);
void cxi_ct_reset_success(struct cxi_ct *ct);
void cxi_ct_reset_failure(struct cxi_ct *ct);

/* CQ operations */
void cxi_cq_init(struct cxi_cq *cq, void *cmds, unsigned int count,
                 void *csr, unsigned int idx);
int cxi_cq_emit_target(struct cxi_cq *cq, union c_cmdu *cmd);
int cxi_cq_emit_dma(struct cxi_cq *cq, struct c_full_dma_cmd *cmd);
void cxi_cq_ring(struct cxi_cq *cq);
```

These functions operate on mmap'd hardware registers and can potentially be called from GPU kernels if the memory is mapped to GPU address space.
