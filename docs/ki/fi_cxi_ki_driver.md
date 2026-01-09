# KI API v4: CXI Kernel Driver Reference

This document is **reference documentation** for existing CXI kernel driver interfaces that KI uses. **No driver changes are required for KI** — all necessary functionality already exists.

For the userspace library that wraps these interfaces, see [libcxi API](fi_cxi_ki_libcxi.md). For the libfabric provider that builds on these, see [Libfabric API](fi_cxi_ki_libfabric.md).

---

## KI-Relevant Existing Functionality

KI leverages these existing driver capabilities:

| Capability | Driver Command | Purpose for KI |
|------------|----------------|----------------|
| CT allocation with writeback | `CXI_OP_CT_ALLOC` | GPU polls writeback buffer |
| CT writeback update | `CXI_OP_CT_WB_UPDATE` | Point writeback to GPU memory |
| CQ allocation with mmap | `CXI_OP_CQ_ALLOC` | GPU emits commands directly |
| GPU memory registration | `CXI_OP_ATU_MAP` + dmabuf hints | Register GPU buffers for DMA |
| PTE configuration | `CXI_OP_PTE_ALLOC`, `CXI_OP_PTE_MAP` | Target-side CT routing |

---

## 1. Driver Command Interface

The CXI driver uses a unified command dispatch model via `enum cxi_command_opcode`. All operations are submitted through a single ioctl with command-specific structures.

### Command Opcode Enumeration

```c
/* From cxi-abi.h */
enum cxi_command_opcode {
    CXI_OP_INVALID,
    CXI_OP_LNI_ALLOC,
    CXI_OP_LNI_FREE,
    CXI_OP_DOMAIN_RESERVE,
    CXI_OP_DOMAIN_ALLOC,
    CXI_OP_DOMAIN_FREE,
    CXI_OP_CP_ALLOC,
    CXI_OP_CQ_ALLOC,
    CXI_OP_CQ_FREE,
    CXI_OP_ATU_MAP,
    CXI_OP_ATU_UNMAP,
    CXI_OP_EQ_ALLOC,
    CXI_OP_PTE_ALLOC,
    CXI_OP_PTE_MAP,
    CXI_OP_CT_ALLOC,
    CXI_OP_CT_FREE,
    CXI_OP_CT_WB_UPDATE,
    CXI_OP_TRIG_CP_ALLOC,      /* Triggered command profile allocation */
    /* ... 100+ total commands */
    CXI_OP_MAX,
};
```

### Common Command Pattern

Each command follows a standard structure pattern:

```c
struct cxi_<operation>_cmd {
    enum cxi_command_opcode op;
    void __user *resp;           /* Response buffer pointer */
    /* Operation-specific fields */
};

struct cxi_<operation>_resp {
    /* Response fields */
};
```

---

## 2. Counting Event (CT) Allocation

CTs provide hardware counters for tracking operation completion. The driver exposes doorbell MMIO for user-space access.

### CT Allocation Command

```c
struct cxi_ct_alloc_cmd {
    enum cxi_command_opcode op;
    void __user *resp;

    unsigned int lni;                       /* LNI to associate with CT */
    struct c_ct_writeback __user *wb;       /* User writeback buffer */
};

struct cxi_ct_alloc_resp {
    unsigned int ctn;                       /* Allocated CT number */
    struct cxi_mminfo doorbell;             /* mmap info for doorbell */
};

struct cxi_mminfo {
    __u64 offset;                           /* mmap offset */
    __u64 size;                             /* mapping size */
};
```

### CT Writeback Structure

The writeback buffer is updated by hardware when CT values change:

```c
struct c_ct_writeback {
    uint64_t ct_success   : 48;  /* Success counter (48 bits) */
    uint8_t  ct_failure   :  7;  /* Failure counter (7 bits) */
    uint16_t unused       :  8;  /* Reserved */
    uint8_t  ct_writeback :  1;  /* Writeback enable flag */
};
```

**Requirements:**
- Writeback buffer must be **8-byte aligned**
- Must be DMA-accessible (pinned user memory or GPU memory)
- Size: 8 bytes

### CT Doorbell MMIO

Each CT has a dedicated doorbell page for triggering operations:

```c
/* Doorbell physical address calculation (from cass_ct.c:418) */
phys_addr_t doorbell_addr = hw->regs_base + C_MEMORG_CQ_TOU +
                            (ctn * C_TOU_LAUNCH_PAGE_SIZE);
size_t doorbell_size = PAGE_SIZE;  /* 4KB per CT */
```

**Kernel API for Userspace Access:**

```c
/* cxi_ct_user_info() - Get doorbell info for userspace mmap */
int cxi_ct_user_info(struct cxi_ct *ct, phys_addr_t *doorbell_addr,
                     size_t *doorbell_size);
```

The driver returns this info via `cxi_ct_alloc_resp.doorbell` (`struct cxi_mminfo`), which userspace libraries use to mmap the doorbell register.

**Hardware Constants:**
- `C_MEMORG_CQ_TOU`: 0x04000000 (64MB into BAR)
- `C_TOU_LAUNCH_PAGE_SIZE`: 4KB
- `C_NUM_CTS`: 2048 maximum CTs

**Userspace Doorbell Operations:**

Once mmap'd, userspace can perform CT operations via direct register writes:

```c
/* Inline functions (typically in cxi_prov_hw.h or equivalent) */
void cxi_ct_inc_success(struct cxi_ct *ct, uint64_t count);
void cxi_ct_inc_failure(struct cxi_ct *ct, uint64_t count);
void cxi_ct_reset_success(struct cxi_ct *ct);
void cxi_ct_reset_failure(struct cxi_ct *ct);
```

These write to specific offsets within the doorbell page to increment/reset CT counters.

### CT Writeback Update

```c
struct cxi_ct_wb_update_cmd {
    enum cxi_command_opcode op;
    unsigned int ctn;
    struct c_ct_writeback __user *wb;       /* New writeback buffer */
};
```

The driver handles atomic writeback buffer update with proper synchronization:
1. Stops writeback writes by zeroing `tou_wb_credits`
2. Waits for pending writebacks to drain
3. Updates writeback address
4. Re-enables writeback credits

---

## 3. GPU Memory Registration

The driver supports two mechanisms for GPU memory registration.

### DMA-BUF (Preferred)

The driver uses `dma_buf_dynamic_attach()` for proper GPU memory handling with migration support.

**User API (via ATU_MAP with hints):**

```c
struct cxi_md_hints {
    int page_shift;
    int huge_shift;
    int ptg_mode;
    bool ptg_mode_valid;
    int dmabuf_fd;                          /* dmabuf file descriptor */
    unsigned long dmabuf_offset;            /* Offset within dmabuf */
    bool dmabuf_valid;                      /* Set true to use dmabuf */
};

struct cxi_atu_map_cmd {
    enum cxi_command_opcode op;
    void __user *resp;
    unsigned int lni;
    __u64 va;
    __u64 len;
    __u32 flags;
    struct cxi_md_hints hints;
};
```

**Driver Implementation (cass_dma_buf.c):**

```c
/* Attach with move notification support */
md_priv->dmabuf_attach = dma_buf_dynamic_attach(
    md_priv->dmabuf,
    &dev->pdev->dev,
    &cxi_dma_buf_attach_ops,      /* Includes move_notify callback */
    md_priv
);

/* Map with DMA addresses */
md_priv->dmabuf_sgt = dma_buf_map_attachment(
    md_priv->dmabuf_attach,
    DMA_BIDIRECTIONAL
);
```

**Move Notification:**

The driver implements `move_notify` to handle GPU memory migration:

```c
static void cxi_dma_buf_move_notify(struct dma_buf_attachment *attach) {
    struct cxi_md_priv *md_priv = attach->importer_priv;
    /* Invalidate ATU entries for migrated memory */
    cass_clear_range(md_priv, md_priv->md.iova, md_priv->md.len);
    cass_invalidate_range(cac, md_priv->md.iova, md_priv->md.len);
}
```

### NVIDIA P2P (Fallback)

For systems without dmabuf support, the driver uses NVIDIA's proprietary P2P API.

**Module Parameter:**

```c
static bool nv_p2p_persistent = true;
module_param(nv_p2p_persistent, bool, 0644);
```

**Implementation (cass_nvidia_gpu.c):**

```c
/* Persistent mode (recommended) - no callback needed */
if (nv_p2p_persistent && p2p_get_pages_pers)
    return p2p_get_pages_pers(va, len, &p2p_info->page_table, 0);

/* Non-persistent mode - requires invalidation callback */
return p2p_get_pages(0, 0, va, len, &p2p_info->page_table,
                     nv_free_callback, md_priv);
```

**DMA Mapping:**

```c
/* Map GPU pages to NIC-accessible addresses */
ret = nvidia_p2p_dma_map_pages(pdev, p2p_info->page_table, &dma_mapping);
```

**Notes:**
- Default page size: 64KB (`NV_DEF_PAGE_SHIFT = 16`)
- Requires NVIDIA kernel symbols at runtime
- Persistent mode avoids invalidation races

---

## 4. Command Queue (CQ) Allocation

CQs are used to submit operations to the NIC. Triggered CQs are required for hardware-triggered operations.

### CQ Flags

```c
enum {
    CXI_CQ_USER              = (1 << 0),  /* User-space CQ */
    CXI_CQ_IS_TX             = (1 << 1),  /* Transmit (vs target) CQ */
    CXI_CQ_TX_ETHERNET       = (1 << 2),  /* Raw Ethernet support */
    CXI_CQ_TX_WITH_TRIG_CMDS = (1 << 3),  /* Reserved for triggered commands */
};
```

### CQ Allocation

```c
struct cxi_cq_alloc_opts {
    unsigned int count;                     /* Number of 64-byte entries */
    enum cxi_cq_update_policy policy;       /* Status update policy */
    unsigned int stat_cnt_pool;
    uint32_t flags;                         /* CXI_CQ_* flags */
    unsigned int lcid;                      /* Initial Local Communication ID */
    unsigned int lpe_cdt_thresh_id;
};

struct cxi_cq_alloc_cmd {
    enum cxi_command_opcode op;
    void __user *resp;
    unsigned int lni;
    unsigned int eq;                        /* Event queue for errors */
    struct cxi_cq_alloc_opts opts;
};

struct cxi_cq_alloc_resp {
    unsigned int cq;
    unsigned int count;                     /* Actual entry count */
    struct cxi_mminfo cmds;                 /* mmap for command buffer */
    struct cxi_mminfo wp_addr;              /* mmap for write pointer */
};
```

### CQ MMIO Addresses

```c
/* TX CQ launch address (from cass_cq.c:66-71) */
phys_addr_t cq_mmio_phys_addr(const struct cass_dev *hw, int cq_id) {
    return hw->regs_base + C_CQ_LAUNCH_TXQ_BASE +
           C_CQ_LAUNCH_PAGE_SIZE * cq_id;
}
```

**Kernel API for Userspace Access:**

```c
/* cxi_cq_user_info() - Get CQ info for userspace mmap */
int cxi_cq_user_info(struct cxi_cq *cmdq,
                     size_t *cmds_size,           /* Queue size in 64-byte blocks */
                     struct page **cmds_pages,    /* Pages to mmap */
                     phys_addr_t *wp_addr,        /* Write pointer CSR */
                     size_t *wp_addr_size);       /* CSR size */
```

The driver returns mmap info via `cxi_cq_alloc_resp.cmds` and `cxi_cq_alloc_resp.wp_addr`, which userspace libraries use to:
1. Map the command buffer for emitting commands
2. Map the write pointer register for doorbell operations

**Userspace CQ Operations:**

Once mmap'd, userspace emits commands directly to the command buffer:

```c
/* Inline functions (typically in cxi_prov_hw.h or equivalent) */
void cxi_cq_init(struct cxi_cq *cq, void *cmds, unsigned int count,
                 void *csr, unsigned int idx);
int cxi_cq_emit_target(struct cxi_cq *cq, union c_cmdu *cmd);
int cxi_cq_emit_dma(struct cxi_cq *cq, struct c_full_dma_cmd *cmd);
void cxi_cq_ring(struct cxi_cq *cq);  /* Ring doorbell to submit commands */
```

**Hardware Limits:**
- TX CQs: 1024 (`C_NUM_TRANSMIT_CQS`)
- Target CQs: 512 (`C_NUM_TARGET_CQS`)
- Max entries: 65536 (`CXI_MAX_CQ_COUNT`)

---

## 5. Triggered Command Profile Allocation

Triggered operations require a special communication profile that reserves LCID space.

### Triggered CP Types

```c
enum cxi_trig_cp {
    TRIG_LCID,      /* Triggered command queue LCID only */
    NON_TRIG_LCID,  /* Non-triggered command queue LCID only */
    ANY_LCID,       /* Either triggered or non-triggered */
};
```

### Allocation Command

```c
struct cxi_trig_cp_alloc_cmd {
    enum cxi_command_opcode op;
    void __user *resp;
    unsigned int lni;
    unsigned int vni;
    enum cxi_traffic_class tc;
    enum cxi_traffic_class_type tc_type;
    enum cxi_trig_cp cp_type;               /* Triggered CP type */
};

struct cxi_cp_alloc_resp {
    unsigned int cp_hndl;
    unsigned int lcid;                      /* Allocated LCID */
};
```

---

## 6. Portal Table Entry (PtlTE) Configuration

PtlTEs receive incoming messages and can be configured for CT counting.

### PTE Allocation Options

```c
struct cxi_pt_alloc_opts {
    uint64_t en_event_match         :  1;   /* Match bit routing */
    uint64_t clr_remote_offset      :  1;
    uint64_t en_flowctrl            :  1;   /* Flow control */
    uint64_t use_long_event         :  1;
    uint64_t lossless               :  1;
    uint64_t en_restricted_unicast_lm : 1;
    uint64_t use_logical            :  1;   /* Logical (vs physical) addressing */
    uint64_t is_matching            :  1;   /* Matching portal */
    uint64_t do_space_check         :  1;
    uint64_t en_align_lm            :  1;
    uint64_t en_sw_hw_st_chng       :  1;
    uint64_t ethernet               :  1;
    uint64_t signal_invalid         :  1;
    uint64_t signal_underflow       :  1;
    uint64_t signal_overflow        :  1;
    uint64_t signal_inexact         :  1;
    uint64_t en_match_on_vni        :  1;   /* VNI-based matching */
};
```

### PTE Mapping

```c
struct cxi_pte_map_cmd {
    enum cxi_command_opcode op;
    void __user *resp;
    unsigned int pte_number;
    unsigned int domain_hndl;
    unsigned int pid_offset;
    bool is_multicast;
};

struct cxi_pte_map_resp {
    unsigned int pte_index;
};
```

### PTE Status

```c
struct cxi_pte_status {
    __u32 drop_count;
    __u8 state;
    __u16 les_reserved;
    __u16 les_allocated;
    __u16 les_max;
    __u64 __user *ule_offsets;
    __u16 ule_count;
};
```

**Hardware Limits:**
- Portal Tables: 2048 (`C_NUM_PTLTES`)
- List Entries: 2048 (`C_NUM_TLES`)
- Processing Elements: 4 (`C_PE_COUNT`)

---

## 7. Memory Descriptor (MD)

MDs describe registered memory regions for NIC access.

```c
struct cxi_md {
    __u64    iova;          /* IO virtual address (NIC-visible) */
    __u64    va;            /* Virtual address */
    size_t   len;           /* Length */
    __u8     lac;           /* Logical address context */
    int      page_shift;    /* Base page size */
    int      huge_shift;    /* Huge page size */
    unsigned int id;
};

/* Helper macros */
#define CXI_VA_TO_IOVA(_md, _va)  ((_md)->iova + ((__u64)(_va) - (_md)->va))
#define CXI_IOVA_TO_VA(_md, _iova) ((_md)->va + ((__u64)(_iova) - (_md)->iova))
```

### ATU Map Flags

```c
enum cxi_atu_map_flags {
    CXI_MAP_PIN       = (1 << 0),   /* Pin pages */
    CXI_MAP_ATS       = (1 << 1),   /* ATS (address translation services) */
    CXI_MAP_WRITE     = (1 << 2),   /* Write access */
    CXI_MAP_READ      = (1 << 3),   /* Read access */
    CXI_MAP_FAULT     = (1 << 4),   /* Fault on access */
    CXI_MAP_NOCACHE   = (1 << 5),   /* Non-cached */
    CXI_MAP_USER_ADDR = (1 << 6),   /* User address */
    CXI_MAP_DEVICE    = (1 << 9),   /* Device memory (GPU) */
    CXI_MAP_ALLOC_MD  = (1 << 11),  /* Allocate MD */
    CXI_MAP_PREFETCH  = (1 << 12),  /* Prefetch pages */
};
```

---

## 8. GPU Memory Mapping Summary

| Direction | Use Case | Approach | Implementation |
|-----------|----------|----------|----------------|
| GPU → NIC | CT writeback, data buffers | dmabuf (preferred) | `cass_dma_buf.c` |
| GPU → NIC | CT writeback, data buffers | NVIDIA P2P (fallback) | `cass_nvidia_gpu.c` |
| NIC → GPU | Doorbell MMIO | Platform-specific | Not in driver |

### dmabuf Requirements

| Platform | Version | API |
|----------|---------|-----|
| NVIDIA | CUDA 11.7+ | `cuMemExportToShareableHandle(..., CU_MEM_HANDLE_TYPE_DMABUF)` |
| AMD | ROCm 5.0+ | `hipMemExportToShareableHandle(..., hipMemHandleTypeDmaBufFd)` |

### NVIDIA P2P Requirements

- Kernel symbols: `nvidia_p2p_get_pages`, `nvidia_p2p_dma_map_pages`, etc.
- Optional: `nvidia_p2p_get_pages_persistent` for persistent mode
- Default page size: 64KB

---

## 9. KI-Specific Requirements

For GPU-initiated networking, the following must be in GPU-accessible memory:

1. **CT Writeback Buffer** - 8-byte aligned, DMA-accessible
2. **Signal Arrays** - For peer notification (Model A/B)

The doorbell MMIO mapping (NIC → GPU) remains platform-specific and is handled in user space.

### CT Writeback for GPU Polling

```c
/* GPU kernel polls this structure */
struct c_ct_writeback *gpu_wb = /* GPU memory */;

/* Wait for completion */
while (gpu_wb->ct_success < expected_count) {
    /* GPU polling loop */
}
```

### Resource Limits

| Resource | Max Count | Notes |
|----------|-----------|-------|
| CTs | 2048 | Per-device |
| TX CQs | 1024 | Per-device |
| Target CQs | 512 | Per-device |
| PTEs | 2048 | Per-device |
| Processing Elements | 4 | LE pools per PE |

---

## API Reference

### User-Kernel Commands (via ioctl)

| Command | Purpose |
|---------|---------|
| `CXI_OP_CT_ALLOC` | Allocate CT with writeback buffer |
| `CXI_OP_CT_WB_UPDATE` | Update CT writeback address |
| `CXI_OP_CT_FREE` | Free counting event |
| `CXI_OP_CQ_ALLOC` | Allocate command queue |
| `CXI_OP_TRIG_CP_ALLOC` | Allocate triggered command profile |
| `CXI_OP_ATU_MAP` | Map memory (with dmabuf hints) |
| `CXI_OP_ATU_UNMAP` | Unmap memory |
| `CXI_OP_PTE_ALLOC` | Allocate portal table entry |
| `CXI_OP_PTE_MAP` | Map PTE to domain |

### Exported Kernel APIs (for libcxi)

| Function | Purpose |
|----------|---------|
| `cxi_ct_alloc()` | Allocate counting event |
| `cxi_ct_user_info()` | Get CT doorbell physical address for mmap |
| `cxi_ct_wb_update()` | Update CT writeback buffer |
| `cxi_ct_free()` | Free counting event |
| `cxi_cq_alloc()` | Allocate command queue |
| `cxi_cq_user_info()` | Get CQ buffer/CSR addresses for mmap |
| `cxi_cq_free()` | Free command queue |
| `cxi_trig_cp_alloc()` | Allocate triggered communication profile |
| `cxi_map()` | Map memory (CPU or GPU via dmabuf) |
| `cxi_unmap()` | Unmap memory descriptor |
| `cxi_pte_alloc()` | Allocate portal table entry |
| `cxi_pte_map()` | Map PTE to domain |

### Userspace Hardware Access (inline functions)

These functions operate on mmap'd memory regions:

| Function | Purpose |
|----------|---------|
| `cxi_ct_inc_success()` | Increment CT success counter via doorbell |
| `cxi_ct_inc_failure()` | Increment CT failure counter via doorbell |
| `cxi_ct_reset_success()` | Reset CT success counter |
| `cxi_ct_reset_failure()` | Reset CT failure counter |
| `cxi_cq_emit_target()` | Emit target (receive) command to CQ |
| `cxi_cq_emit_dma()` | Emit DMA command to CQ |
| `cxi_cq_ring()` | Ring CQ doorbell to submit commands |

---

## Source Files

| Component | File | Key Functions |
|-----------|------|---------------|
| CT management | `cass_ct.c` | `cxi_ct_alloc()`, `cxi_ct_user_info()`, `cxi_ct_wb_update()` |
| CQ management | `cass_cq.c` | `cass_cq_init()`, `setup_hw_tx_cq()` |
| PTE management | `cass_pt.c` | `cass_pte_init()`, `init_le_pools()` |
| dmabuf support | `cass_dma_buf.c` | `cxi_dmabuf_get_pages()`, `cxi_dmabuf_put_pages()` |
| NVIDIA P2P | `cass_nvidia_gpu.c` | `nvidia_get_pages()`, `nvidia_put_pages()` |
| Command dispatch | `cxi_user_core.c` | Command handler table and dispatch |
| Public ABI | `cxi-abi.h` | All user-kernel interface definitions |
