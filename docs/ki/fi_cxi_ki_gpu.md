# KI API v4: GPU-Side Interface

Header: `fi_cxi_ki_gpu.h`

This document describes the GPU-side header-only library for posting RDMA operations from GPU kernels. For host-side setup and metadata preparation, see [Libfabric API](fi_cxi_ki_libfabric.md).

---

## Summary: What's New

This is **entirely new code** — a header-only library that GPU kernels include.

### Core Functions

| Function | Purpose |
|----------|---------|
| `fi_cxi_inject_put*` | Build and emit PUT commands to mmap'd CQ |
| `fi_cxi_ring_doorbell` | Write to mmap'd CSR to notify NIC |
| `fi_cxi_cntr_*` | Poll/reset local counters (send completion) |
| `fi_cxi_target_ct_*` | Poll/reset target CTs (receive completion, Model A) |
| `fi_cxi_signal_*` | Poll/reset signals (Model B) |

### Dependencies

The GPU header requires:
- `fi_ki_meta_t` metadata prepared by host via `fi_cxi_ki_ops`
- Command queue buffer and write pointer mmap'd to GPU address space
- CT writeback buffer in GPU-accessible memory

---

## Constants

```c
typedef enum {
    FI_KI_COOP_THREAD = 0,
    FI_KI_COOP_WARP   = 1,
    FI_KI_COOP_BLOCK  = 2,
} fi_ki_coop_t;

#define FI_KI_NO_SIGNAL   UINT32_MAX
#define FI_KI_NO_COUNTER  UINT32_MAX
#define FI_KI_NO_TAG      0ULL
```

---

## Cooperative Modes

The `fi_ki_coop_t` parameter specifies how multiple GPU threads coordinate when posting operations.

### FI_KI_COOP_THREAD (Default)

Each thread operates independently. Thread-safe but may have contention on write pointer.

```c
/* Each thread allocates its own command slot */
void *cmd = ki_alloc_cmd_slots(meta, ctx, 2);  /* Uses atomicAdd internally */
```

### FI_KI_COOP_WARP

All threads in a warp cooperatively post a single operation. One thread (lane 0) handles the command queue interaction.

```c
__device__ int fi_cxi_inject_put_warp(
    fi_ki_meta_t meta, int ctx, int peer_idx,
    void const *src, uint64_t dst_off, size_t len
) {
    int lane = threadIdx.x & 31;

    /* Only lane 0 allocates and posts */
    if (lane == 0) {
        void *cmd = ki_alloc_dma_cmd(meta, ctx);
        /* ... build command ... */
    }

    /* All lanes synchronize before returning */
    __syncwarp();
    return 0;
}
```

Use when: A single operation is computed cooperatively by the warp.

### FI_KI_COOP_BLOCK

All threads in a CTA/block cooperate. One thread (typically thread 0) handles doorbell ringing.

```c
__global__ void moe_dispatch(fi_ki_meta_t meta, ...) {
    int ctx = blockIdx.x % meta->context_count;

    /* Each thread posts its operations */
    for (...) {
        fi_cxi_inject_put_simple(meta, ctx, peer, src, dst_off, len,
                                  FI_KI_COOP_THREAD, counter);
    }

    /* Synchronize block before ringing doorbell */
    __syncthreads();

    /* Only thread 0 rings the doorbell for the block */
    if (threadIdx.x == 0) {
        fi_cxi_ring_doorbell(meta, ctx);
    }

    __syncthreads();
}
```

Use when: Multiple threads in a block contribute commands to the same context.

### Doorbell Semantics by Mode

| Mode | Who Rings Doorbell | When |
|------|-------------------|------|
| `THREAD` | Each thread | After its operations |
| `WARP` | Lane 0 | After warp's operation |
| `BLOCK` | Thread 0 | After `__syncthreads()` |

### Context Assignment Patterns

**Per-CTA context (recommended):**
```c
int ctx = blockIdx.x % meta->context_count;
```

**Per-warp context (high throughput):**
```c
int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
int ctx = warp_id % meta->context_count;
```

**Per-thread context (maximum parallelism, high resource usage):**
```c
int ctx = threadIdx.x % meta->context_count;
```

---

## Return Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `-FI_EAGAIN` | Queue full (ring doorbell and retry) |
| `-FI_EINVAL` | Invalid parameter |
| `-FI_EIO` | Hardware/link error |

---

## Error Handling

### Queue Full Condition

When the command queue is full, `ki_alloc_cmd_slots()` may wrap around and overwrite pending commands. To prevent this:

```c
/* Safe command allocation with backpressure */
__device__ void* ki_alloc_cmd_safe(fi_ki_meta_t meta, int ctx, uint32_t slots) {
    struct fi_cxi_ki_cmdq_config *cfg = &meta->cmdq_config[ctx];
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];

    /* Wait until there's room in the queue */
    while (true) {
        uint32_t wp = hot->wp;
        uint32_t completed = fi_cxi_cntr_read(meta, ctx);  /* Bound counter */
        uint32_t in_flight = wp - completed;

        if (in_flight + slots <= cfg->mask) {
            uint32_t slot = atomicAdd((uint32_t*)&hot->wp, slots);
            return (char*)cfg->buf + (slot & cfg->mask) * cfg->slot_size;
        }

        /* Queue full - wait for completions */
        __nanosleep(100);
    }
}
```

### Counter Overflow

Counters are 48 bits for success and 7 bits for failure. Use rolling comparison to handle wrap:

```c
/* Correct threshold comparison (handles 48-bit wrap) */
bool threshold_met = ((int64_t)(current - threshold) >= 0);

/* The counter will wrap after 2^48 operations (~281 trillion).
 * As long as threshold checks happen frequently relative to wrap,
 * rolling comparison remains correct. */
```

### NIC Errors

Hardware errors are reported via the failure counter. Always check for failures when waiting:

```c
__device__ int fi_cxi_cntr_wait_safe(fi_ki_meta_t meta, uint32_t idx, uint64_t threshold) {
    while (1) {
        uint64_t val = fi_cxi_cntr_read(meta, idx);
        if ((int64_t)(val - threshold) >= 0) return 0;

        /* Check for hardware errors */
        uint64_t failures = fi_cxi_cntr_read_failure(meta, idx);
        if (failures > 0) {
            /* Error occurred - operation failed.
             * Host should query EQ for details. */
            return -FI_EIO;
        }

        __nanosleep(100);
    }
}
```

### Link Errors and Recovery

Link errors (cable disconnect, NIC reset) require host intervention:

1. GPU detects via persistent failure counter increment
2. GPU returns `-FI_EIO` to caller
3. Host queries event queue for error details
4. Host initiates recovery (reconnect, re-register memory, etc.)
5. Host signals GPU to resume or abort

The GPU cannot recover from link errors autonomously.

---

## Metadata Handle

The `fi_ki_meta_t` type is an opaque handle to metadata prepared by the host. See [Libfabric API: Metadata Structure](fi_cxi_ki_libfabric.md#metadata-structure-definition) for the internal layout.

```c
typedef struct fi_cxi_ki_meta *fi_ki_meta_t;
```

---

## PUT Operations

### fi_cxi_inject_put (Model B: With Initiator Signals)

POST a PUT with optional sender-triggered remote signal:

```c
__device__ int fi_cxi_inject_put(
    fi_ki_meta_t  meta,
    int           context_index,
    int           peer_index,
    const void   *local_buf,
    uint64_t      remote_offset,
    size_t        length,
    fi_ki_coop_t  coop,
    uint32_t      remote_signal_idx,
    uint64_t      remote_signal_value,
    uint32_t      local_counter_idx
);
```

Use when sender needs to explicitly notify receiver.

### fi_cxi_inject_put_tagged (Model A: For Target CT Routing)

POST a PUT with match bits for receiver-side CT routing:

```c
__device__ int fi_cxi_inject_put_tagged(
    fi_ki_meta_t  meta,
    int           context_index,
    int           peer_index,
    const void   *local_buf,
    uint64_t      remote_offset,
    size_t        length,
    uint64_t      match_bits,         /* Tag for receiver CT routing */
    fi_ki_coop_t  coop,
    uint32_t      local_counter_idx
);
```

The `match_bits` value is used by the receiver's NIC to route to a specific target CT (when configured with `FI_CXI_KI_TARGET_CT_PER_PEER` or `FI_CXI_KI_TARGET_CT_MATCH_BITS`).

For per-peer tracking, use `match_bits = my_rank` (sender's rank).

### fi_cxi_inject_put_simple (Model A: Aggregate Target CT)

POST a PUT without match bits (receiver uses aggregate CT):

```c
__device__ int fi_cxi_inject_put_simple(
    fi_ki_meta_t  meta,
    int           context_index,
    int           peer_index,
    const void   *local_buf,
    uint64_t      remote_offset,
    size_t        length,
    fi_ki_coop_t  coop,
    uint32_t      local_counter_idx
);
```

Equivalent to `fi_cxi_inject_put` with `FI_KI_NO_SIGNAL`.

---

## Doorbell and Flush

### fi_cxi_ring_doorbell

Notifies the NIC that new commands have been written to the command queue:

```c
__device__ void fi_cxi_ring_doorbell(
    fi_ki_meta_t meta,
    int          context_index
);
```

**Implementation:**
```c
__device__ void fi_cxi_ring_doorbell(fi_ki_meta_t meta, int ctx) {
    FI_KI_FENCE_SYSTEM();  /* Ensure commands visible to NIC (see Memory Ordering) */
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];
    *hot->wp_addr = hot->wp;  /* Write to mmap'd write pointer CSR */
}
```

After calling `fi_cxi_ring_doorbell`, the NIC begins processing commands. The function returns immediately — it does not wait for command completion.

### fi_cxi_flush

Waits until the NIC has **consumed** all commands from the command queue (not completed — just read from the queue). This is useful when the GPU needs to reuse command buffer slots.

```c
__device__ void fi_cxi_flush(
    fi_ki_meta_t meta,
    int          context_index,
    fi_ki_coop_t coop
);
```

**When to use flush:**
- Before overwriting command buffer entries that may still be in-flight
- When the command queue is nearly full and you need to wait for slots

**When NOT to use flush:**
- To wait for data transfer completion — use `fi_cxi_cntr_wait` (send) or `fi_cxi_target_ct_wait` (receive) instead
- After every command — this defeats the purpose of batching

### Flush Implementation Options

Flush can be implemented several ways depending on hardware capabilities:

**Option 1: Poll CQ Status (if available)**

If the CQ has a hardware-updated status field showing consumed entries:

```c
__device__ void fi_cxi_flush(fi_ki_meta_t meta, int ctx, fi_ki_coop_t coop) {
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];

    /* Wait until NIC's read pointer catches up to our write pointer */
    while (hot->status->rp != hot->wp) {
        __nanosleep(100);
    }
}
```

**Option 2: Use Local Counter**

Bind a counter to operations and wait for it to increment:

```c
__device__ void fi_cxi_flush(fi_ki_meta_t meta, int ctx, fi_ki_coop_t coop) {
    /* All commands posted with local_counter_idx bound will increment CT */
    uint64_t expected = meta->cmdq_hot[ctx].cmds_posted;
    fi_cxi_cntr_wait(meta, ctx, expected);
}
```

**Option 3: Implicit via Queue Depth**

Track posted vs completed commands and block when queue is full:

```c
__device__ void* ki_alloc_cmd(fi_ki_meta_t meta, int ctx) {
    struct fi_cxi_ki_cmdq_config *cfg = &meta->cmdq_config[ctx];
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];

    /* If queue is full, wait for completions */
    while ((hot->wp - hot->completed) >= cfg->size) {
        /* Poll completion counter or status */
        __nanosleep(100);
    }

    uint32_t slot = atomicAdd(&hot->wp, 1) & cfg->mask;
    return (char*)cfg->buf + slot * cfg->cmd_size;
}
```

### Flush vs Counter Wait

| Operation | Waits For | Use Case |
|-----------|-----------|----------|
| `fi_cxi_flush` | NIC consumed command from queue | Reuse command buffer slots |
| `fi_cxi_cntr_wait` | DMA completed, source buffer safe | Reuse source data buffer |
| `fi_cxi_target_ct_wait` | Data arrived at receiver | Process received data |

**Note:** Command consumption (flush) happens before DMA completion. A command can be consumed from the queue while the DMA is still in progress.

---

## Local Counter Operations (Send Completion)

Track when source buffers are safe to reuse:

```c
__device__ uint64_t fi_cxi_cntr_read(fi_ki_meta_t meta, uint32_t counter_idx);
__device__ uint64_t fi_cxi_cntr_read_failure(fi_ki_meta_t meta, uint32_t counter_idx);
__device__ int fi_cxi_cntr_wait(fi_ki_meta_t meta, uint32_t counter_idx,
                                 uint64_t threshold);
__device__ void fi_cxi_cntr_reset(fi_ki_meta_t meta, uint32_t counter_idx);
```

---

## Target CT Operations (Model A: Receive Completion)

Track incoming PUTs on the receiver side:

```c
/* Read current success count (acquire semantics) */
__device__ uint64_t fi_cxi_target_ct_read(
    fi_ki_meta_t meta,
    uint32_t     ct_idx
);

/* Read current failure count */
__device__ uint64_t fi_cxi_target_ct_read_failure(
    fi_ki_meta_t meta,
    uint32_t     ct_idx
);

/* Wait for success count >= threshold (blocking, acquire semantics) */
__device__ int fi_cxi_target_ct_wait(
    fi_ki_meta_t meta,
    uint32_t     ct_idx,
    uint64_t     threshold
);

/* Reset CT to zero (must not race with incoming PUTs) */
__device__ void fi_cxi_target_ct_reset(
    fi_ki_meta_t meta,
    uint32_t     ct_idx
);
```

**Aggregate mode** (`FI_CXI_KI_TARGET_CT_AGGREGATE`): Use `ct_idx = 0` for all operations.

**Per-peer mode** (`FI_CXI_KI_TARGET_CT_PER_PEER`): Use `ct_idx = peer_rank` to wait for specific peer.

---

## Signal Operations (Model B: Initiator-Triggered)

Poll memory that peers increment via FAA:

```c
__device__ uint64_t fi_cxi_signal_read(fi_ki_meta_t meta, uint32_t signal_idx);
__device__ int fi_cxi_signal_wait(fi_ki_meta_t meta, uint32_t signal_idx,
                                   uint64_t threshold);
__device__ void fi_cxi_signal_reset(fi_ki_meta_t meta, uint32_t signal_idx);
```

### fi_cxi_signal_send

Send signal without data (for barriers, notifications):

```c
__device__ int fi_cxi_signal_send(
    fi_ki_meta_t meta,
    int          context_index,
    int          peer_index,
    uint32_t     remote_signal_idx,
    uint64_t     value
);
```

---

## Memory Ordering

### Platform-Specific Fence Functions

The GPU header uses platform-specific memory ordering primitives. Implementations must provide:

| Operation | CUDA | HIP | SYCL |
|-----------|------|-----|------|
| System fence | `__threadfence_system()` | `__threadfence_system()` | `atomic_fence(seq_cst, system)` |
| Device fence | `__threadfence()` | `__threadfence()` | `atomic_fence(seq_cst, device)` |
| Atomic load | `__atomic_load_n(..., __ATOMIC_ACQUIRE)` | same | `atomic_load_explicit(..., acquire)` |
| Atomic store | `__atomic_store_n(..., __ATOMIC_RELEASE)` | same | `atomic_store_explicit(..., release)` |

**Portability Macros:**
```c
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    #define FI_KI_FENCE_SYSTEM()  __threadfence_system()
    #define FI_KI_FENCE_DEVICE()  __threadfence()
#elif defined(__SYCL_DEVICE_ONLY__)
    #define FI_KI_FENCE_SYSTEM()  sycl::atomic_fence(sycl::memory_order::seq_cst, \
                                                      sycl::memory_scope::system)
    #define FI_KI_FENCE_DEVICE()  sycl::atomic_fence(sycl::memory_order::seq_cst, \
                                                      sycl::memory_scope::device)
#endif
```

### Acquire/Release Semantics

All read operations have **acquire** semantics.
All write/reset operations have **release** semantics.

### Target CT Visibility Guarantee

When `fi_cxi_target_ct_read()` returns a value >= N, all data from the first N PUTs to that CT is visible.

### Signal Visibility Guarantee

When `fi_cxi_signal_read()` returns a value >= N, all data from PUTs that triggered those signal increments is visible.

### Ordering Within Context

PUTs on the same context to the same peer complete in order.

---

## Rolling Comparison

Target CTs, counters, and signals use 64-bit values with rolling comparison:

```c
bool threshold_met = ((int64_t)(current - threshold) >= 0);
```

---

## Implementation Sketches

The following shows how GPU-side functions access the SoA metadata structure. These are illustrative implementations; actual implementations may vary based on CXI command formats.

### Command Queue Access

```c
/* Allocate command slots in the queue
 *
 * CXI commands use 32-byte base slots:
 * - Simple commands: 1 slot (32 bytes)
 * - DMA PUT/GET: 2 slots (64 bytes)
 * - Triggered operations: 4 slots (128 bytes)
 *
 * The write pointer tracks 32-byte slots.
 */
__device__ static inline void* ki_alloc_cmd_slots(
    fi_ki_meta_t meta,
    int ctx,
    uint32_t num_slots  /* 1, 2, or 4 */
) {
    struct fi_cxi_ki_cmdq_config *cfg = &meta->cmdq_config[ctx];
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];

    /* Atomically reserve slots */
    uint32_t wp = atomicAdd((uint32_t*)&hot->wp, num_slots);
    uint32_t slot = wp & cfg->mask;
    return (char*)cfg->buf + slot * cfg->slot_size;
}

/* Convenience: allocate single 32-byte slot */
__device__ static inline void* ki_alloc_cmd(
    fi_ki_meta_t meta,
    int ctx
) {
    return ki_alloc_cmd_slots(meta, ctx, 1);
}

/* Convenience: allocate 64-byte command (2 slots) for DMA operations */
__device__ static inline void* ki_alloc_dma_cmd(
    fi_ki_meta_t meta,
    int ctx
) {
    return ki_alloc_cmd_slots(meta, ctx, 2);
}

/* Ring doorbell after posting commands */
__device__ static inline void ki_ring_doorbell(
    fi_ki_meta_t meta,
    int ctx
) {
    FI_KI_FENCE_SYSTEM();  /* Ensure commands visible to NIC */
    struct fi_cxi_ki_cmdq_hot *hot = &meta->cmdq_hot[ctx];
    *hot->wp_addr = hot->wp;
}
```

### PUT Command Building (SoA Peer Access)

```c
/* Build and post a simple PUT command (Model A aggregate) */
__device__ int fi_cxi_inject_put_simple(
    fi_ki_meta_t meta,
    int ctx, int peer_idx,
    void const *src, uint64_t dst_off, size_t len,
    fi_ki_coop_t coop,
    uint32_t counter_idx
) {
    /* Allocate 2 slots (64 bytes) for DMA PUT command */
    void *cmd = ki_alloc_dma_cmd(meta, ctx);

    /* SoA peer access - coalesced when warp threads use consecutive peer_idx.
     * When 32 threads each load from consecutive peer indices, the GPU memory
     * controller combines these into 1-2 cache line fetches per array. */
    uint64_t dfa       = FI_KI_PEER_DFA(meta, peer_idx);
    uint8_t  dfa_ext   = FI_KI_PEER_DFA_EXT(meta, peer_idx);
    uint32_t index_ext = FI_KI_PEER_INDEX_EXT(meta, peer_idx);
    uint64_t mr_base   = FI_KI_PEER_MR_BASE(meta, peer_idx);
    uint64_t mr_key    = FI_KI_PEER_MR_KEY(meta, peer_idx);

    /* Compute destination IOVA */
    uint64_t dst_iova = mr_base + dst_off;

    /* Build CXI DMA PUT command (pseudocode - actual format is HW-specific) */
    struct cxi_put_cmd *put = (struct cxi_put_cmd *)cmd;
    put->opcode = CXI_OP_PUT;
    put->dfa = dfa;
    put->dfa_ext = dfa_ext;
    put->index_ext = index_ext;
    put->lac = meta->config.local_lac;
    put->dst_iova = dst_iova;
    put->length = len;
    put->remote_key = mr_key;

    /* Bind local counter if requested */
    if (counter_idx != FI_KI_NO_COUNTER) {
        put->ct_idx = meta->wb.cntr_ct_idx[counter_idx];
        put->ct_enable = 1;
    }

    return 0;
}

/* Build and post a tagged PUT command (Model A per-peer) */
__device__ int fi_cxi_inject_put_tagged(
    fi_ki_meta_t meta,
    int ctx, int peer_idx,
    void const *src, uint64_t dst_off, size_t len,
    uint64_t match_bits,
    fi_ki_coop_t coop,
    uint32_t counter_idx
) {
    /* Allocate 2 slots (64 bytes) for DMA PUT command */
    void *cmd = ki_alloc_dma_cmd(meta, ctx);

    /* SoA peer access */
    uint64_t dfa     = FI_KI_PEER_DFA(meta, peer_idx);
    uint64_t mr_base = FI_KI_PEER_MR_BASE(meta, peer_idx);
    uint64_t mr_key  = FI_KI_PEER_MR_KEY(meta, peer_idx);

    /* Build PUT command with match bits */
    struct cxi_put_cmd *put = (struct cxi_put_cmd *)cmd;
    /* ... similar to simple PUT, using mr_base + dst_off ... */
    put->match_bits = match_bits;  /* Embedded in command for receiver routing */

    return 0;
}

/* Build and post PUT with initiator signal (Model B)
 *
 * Implementation Options:
 *
 * Option 1: GPU emits PUT + Triggered FAA (recommended for flexibility)
 *   - GPU allocates 2 command slots (PUT) + 4 slots (triggered FAA) = 6 slots
 *   - GPU reads current CT value, sets triggered threshold = current + 1
 *   - Commands processed in order: triggered FAA registered, then PUT executes
 *   - When PUT completes and CT increments, triggered FAA fires
 *
 * Option 2: Host pre-configures triggered FAAs (simpler but less flexible)
 *   - Host uses fi_control(FI_QUEUE_WORK) to queue N triggered FAAs
 *   - Each FAA has threshold = 1, 2, 3, ... N
 *   - GPU just emits PUTs that increment the CT
 *   - Limitation: must know max signals in advance, signal_idx maps to threshold
 *
 * This implementation uses Option 1 for maximum flexibility.
 */
__device__ int fi_cxi_inject_put(
    fi_ki_meta_t meta,
    int ctx, int peer_idx,
    void const *src, uint64_t dst_off, size_t len,
    fi_ki_coop_t coop,
    uint32_t sig_idx, uint64_t sig_val,
    uint32_t counter_idx
) {
    /* SoA peer access - coalesced when warp threads access consecutive peers */
    uint64_t dfa         = FI_KI_PEER_DFA(meta, peer_idx);
    uint8_t  dfa_ext     = FI_KI_PEER_DFA_EXT(meta, peer_idx);
    uint32_t index_ext   = FI_KI_PEER_INDEX_EXT(meta, peer_idx);
    uint64_t mr_base     = FI_KI_PEER_MR_BASE(meta, peer_idx);
    uint64_t mr_key      = FI_KI_PEER_MR_KEY(meta, peer_idx);

    /* If signal requested, emit triggered FAA BEFORE the PUT */
    if (sig_idx != FI_KI_NO_SIGNAL) {
        uint64_t signal_base = FI_KI_PEER_SIGNAL_BASE(meta, peer_idx);
        uint64_t signal_key  = FI_KI_PEER_SIGNAL_KEY(meta, peer_idx);

        /* Read current CT value for threshold calculation */
        uint16_t ct_idx = meta->wb.cntr_ct_idx[counter_idx];
        uint64_t current_ct = fi_cxi_cntr_read(meta, counter_idx);

        /* Allocate 4 slots (128 bytes) for triggered FAA command */
        void *trig_cmd = ki_alloc_cmd_slots(meta, ctx, 4);

        /* Build triggered FAA: fires when CT >= current + 1 */
        struct c_trig_dma_amo_cmd *trig = (struct c_trig_dma_amo_cmd *)trig_cmd;
        trig->command.opcode = C_CMD_CT_TRIG_DMA;
        trig->ct = ct_idx;
        trig->threshold = current_ct + 1;
        trig->dfa = dfa;
        trig->remote_offset = signal_base + sig_idx * sizeof(uint64_t);
        trig->atomic_op = C_AMO_OP_SUM;  /* FAA */
        trig->op1_word1 = sig_val;       /* Value to add */
        /* ... other fields ... */
    }

    /* Allocate 2 slots (64 bytes) for PUT command */
    void *cmd = ki_alloc_cmd_slots(meta, ctx, 2);

    /* Build PUT command */
    struct c_full_dma_cmd *put = (struct c_full_dma_cmd *)cmd;
    put->command.opcode = C_CMD_PUT;
    put->dfa = dfa;
    put->index_ext = index_ext;
    put->lac = meta->config.local_lac;
    put->remote_offset = mr_base + dst_off;
    put->request_len = len;
    put->eq = C_EQ_NONE;  /* No event needed if using CT */

    /* Bind local counter if requested */
    if (counter_idx != FI_KI_NO_COUNTER) {
        put->event_ct_ack = 1;
        put->ct = meta->wb.cntr_ct_idx[counter_idx];
    }

    return 0;
}

/* Warp-coalesced multi-peer PUT (demonstrates SoA benefit) */
__device__ void fi_cxi_put_to_peers_coalesced(
    fi_ki_meta_t meta,
    int ctx,
    void const *src,
    uint64_t dst_off,
    size_t len,
    int base_peer,
    int num_peers  /* should be <= 32 for single warp */
) {
    int lane = threadIdx.x & 31;

    if (lane < num_peers) {
        int peer_idx = base_peer + lane;

        /* All these loads are coalesced across the warp:
         * - Thread 0 loads dfa[base_peer+0], Thread 1 loads dfa[base_peer+1], ...
         * - Memory controller combines into 1-2 cache line fetches
         */
        uint64_t dfa       = FI_KI_PEER_DFA(meta, peer_idx);
        uint64_t dst_iova  = FI_KI_PEER_MR_BASE(meta, peer_idx) + dst_off;
        uint64_t rkey      = FI_KI_PEER_MR_KEY(meta, peer_idx);
        uint8_t  dfa_ext   = FI_KI_PEER_DFA_EXT(meta, peer_idx);
        uint32_t index_ext = FI_KI_PEER_INDEX_EXT(meta, peer_idx);

        /* Post PUT for this peer */
        void *cmd = ki_alloc_cmd(meta, ctx);
        /* ... build command using loaded values ... */
    }
}
```

### Counter and CT Access (Direct Contiguous Arrays)

KI requires batch allocation of counters and target CTs, which places all
writeback buffers in contiguous GPU memory. This enables direct array access
without pointer indirection, improving cache efficiency.

```c
/* Read local counter (send completion) - direct array access */
__device__ uint64_t fi_cxi_cntr_read(fi_ki_meta_t meta, uint32_t idx) {
    /* Contiguous array: meta->wb.cntr_wb[idx] is the c_ct_writeback struct */
    volatile struct c_ct_writeback *wb = &meta->wb.cntr_wb[idx];
    uint64_t raw = __atomic_load_n((uint64_t*)wb, __ATOMIC_ACQUIRE);
    return raw & FI_CXI_CNTR_SUCCESS_MAX;  /* Extract 48-bit success field */
}

__device__ uint64_t fi_cxi_cntr_read_failure(fi_ki_meta_t meta, uint32_t idx) {
    volatile struct c_ct_writeback *wb = &meta->wb.cntr_wb[idx];
    uint64_t raw = __atomic_load_n((uint64_t*)wb, __ATOMIC_ACQUIRE);
    return (raw >> 48) & FI_CXI_CNTR_FAILURE_MAX;  /* Extract 7-bit failure field */
}

__device__ int fi_cxi_cntr_wait(fi_ki_meta_t meta, uint32_t idx, uint64_t threshold) {
    while (1) {
        uint64_t val = fi_cxi_cntr_read(meta, idx);
        if ((int64_t)(val - threshold) >= 0) return 0;

        if (fi_cxi_cntr_read_failure(meta, idx) > 0) return -FI_EIO;

        __nanosleep(100);
    }
}

/* Read target CT (receive completion, Model A) - direct array access */
__device__ uint64_t fi_cxi_target_ct_read(fi_ki_meta_t meta, uint32_t idx) {
    volatile struct c_ct_writeback *wb = &meta->wb.tgt_ct_wb[idx];
    uint64_t raw = __atomic_load_n((uint64_t*)wb, __ATOMIC_ACQUIRE);
    return raw & FI_CXI_CNTR_SUCCESS_MAX;
}

__device__ uint64_t fi_cxi_target_ct_read_failure(fi_ki_meta_t meta, uint32_t idx) {
    volatile struct c_ct_writeback *wb = &meta->wb.tgt_ct_wb[idx];
    uint64_t raw = __atomic_load_n((uint64_t*)wb, __ATOMIC_ACQUIRE);
    return (raw >> 48) & FI_CXI_CNTR_FAILURE_MAX;
}

__device__ int fi_cxi_target_ct_wait(fi_ki_meta_t meta, uint32_t idx, uint64_t threshold) {
    while (1) {
        uint64_t val = fi_cxi_target_ct_read(meta, idx);
        if ((int64_t)(val - threshold) >= 0) return 0;

        if (fi_cxi_target_ct_read_failure(meta, idx) > 0) return -FI_EIO;

        __nanosleep(100);
    }
}

__device__ void fi_cxi_target_ct_reset(fi_ki_meta_t meta, uint32_t idx) {
    /* Reset via MMIO doorbell, not by writing to writeback buffer.
     * The writeback buffer is read-only from GPU perspective - NIC writes it. */
    void *mmio = (char*)meta->wb.cntr_mmio_base + idx * meta->wb.cntr_mmio_stride;
    volatile uint64_t *reset_addr = (uint64_t*)mmio + 16;  /* Reset offset */
    *reset_addr = 0;
}
```

### Signal Access (Model B - Contiguous Array)

Signals are a contiguous array in GPU memory. Remote peers perform FAA operations
to increment signal values. Like counters and target CTs, signals use direct array
access.

```c
/* Read local signal (peers have FAA'd here) - DIRECT array access */
__device__ uint64_t fi_cxi_signal_read(fi_ki_meta_t meta, uint32_t idx) {
    /* meta->wb.signals is a contiguous array, not pointer-to-pointer */
    return __atomic_load_n(&meta->wb.signals[idx], __ATOMIC_ACQUIRE);
}

__device__ int fi_cxi_signal_wait(fi_ki_meta_t meta, uint32_t idx, uint64_t threshold) {
    while (1) {
        uint64_t val = fi_cxi_signal_read(meta, idx);
        if ((int64_t)(val - threshold) >= 0) return 0;
        __nanosleep(100);
    }
}

__device__ void fi_cxi_signal_reset(fi_ki_meta_t meta, uint32_t idx) {
    __atomic_store_n(&meta->wb.signals[idx], 0, __ATOMIC_RELEASE);
}

/* Send signal to peer (no data, Model B)
 * Uses atomic FAA to increment remote signal array.
 */
__device__ int fi_cxi_signal_send(
    fi_ki_meta_t meta,
    int ctx, int peer_idx,
    uint32_t sig_idx, uint64_t value
) {
    /* Allocate 2 slots (64 bytes) for atomic command */
    void *cmd = ki_alloc_dma_cmd(meta, ctx);

    /* SoA peer access - coalesced when warp threads access consecutive peers */
    uint64_t dfa         = FI_KI_PEER_DFA(meta, peer_idx);
    uint64_t signal_base = FI_KI_PEER_SIGNAL_BASE(meta, peer_idx);
    uint64_t signal_key  = FI_KI_PEER_SIGNAL_KEY(meta, peer_idx);

    /* Build atomic FAA command */
    struct cxi_atomic_cmd *atom = (struct cxi_atomic_cmd *)cmd;
    atom->opcode = CXI_OP_ATOMIC_FAA;
    atom->dfa = dfa;
    atom->remote_iova = signal_base + sig_idx * sizeof(uint64_t);
    atom->remote_key = signal_key;
    atom->operand = value;

    return 0;
}
```

---

## Complete Examples

### Example 1: MoE Dispatch with Aggregate Target CT (Simplest)

**Sender (dispatcher):**
```c
__global__ void moe_dispatch(
    fi_ki_meta_t meta,
    half const* tokens,
    uint32_t const* expert_assignments,
    uint64_t const* peer_offsets,
    uint32_t num_tokens,
    uint32_t token_size
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t ctx = tid % 4;

    for (uint32_t t = tid; t < num_tokens; t += blockDim.x * gridDim.x) {
        uint32_t expert = expert_assignments[t];
        uint32_t peer = expert / EXPERTS_PER_RANK;
        uint32_t slot = atomicAdd(&slot_counters[peer], 1);

        void const* src = tokens + t * token_size;
        uint64_t dst_off = peer_offsets[peer] + slot * token_size;

        // Simple PUT - no signal needed, receiver uses target CT
        fi_cxi_inject_put_simple(meta, ctx, peer, src, dst_off, token_size,
                                  FI_KI_COOP_THREAD, ctx);
    }

    __syncthreads();
    if (threadIdx.x < 4) fi_cxi_ring_doorbell(meta, threadIdx.x);
}
```

**Receiver (expert):**
```c
__global__ void moe_receive(
    fi_ki_meta_t meta,
    uint32_t total_expected
) {
    // Wait for all tokens to arrive (aggregate CT)
    if (threadIdx.x == 0) {
        fi_cxi_target_ct_wait(meta, 0, total_expected);
    }
    __syncthreads();

    // Process tokens...

    // Reset for next iteration
    if (threadIdx.x == 0) {
        fi_cxi_target_ct_reset(meta, 0);
    }
}
```

### Example 2: MoE with Per-Peer Tracking

**Sender:**
```c
__global__ void dispatch_tagged(
    fi_ki_meta_t meta,
    uint32_t my_rank,
    ...
) {
    // Use my_rank as match_bits so receiver knows who sent this
    fi_cxi_inject_put_tagged(meta, ctx, peer, src, dst_off, len,
                              my_rank,          /* match_bits */
                              FI_KI_COOP_THREAD, ctx);
}
```

**Receiver:**
```c
__global__ void receive_per_peer(
    fi_ki_meta_t meta,
    uint32_t const* expected_per_peer,
    uint32_t num_peers
) {
    if (threadIdx.x == 0) {
        // Wait for each peer separately
        for (uint32_t p = 0; p < num_peers; p++) {
            if (expected_per_peer[p] > 0) {
                fi_cxi_target_ct_wait(meta, p, expected_per_peer[p]);
            }
        }
    }
    __syncthreads();
}
```

### Example 3: Barrier Using Signals (Model B)

Signals are useful for explicit sender-controlled notification and barriers:

```c
/* Simple barrier - all ranks notify all other ranks */
__global__ void barrier(
    fi_ki_meta_t meta,
    uint32_t my_rank,
    uint32_t num_peers,
    uint64_t round
) {
    if (threadIdx.x == 0) {
        int ctx = 0;

        // Send signal to all peers
        for (uint32_t p = 0; p < num_peers; p++) {
            fi_cxi_signal_send(meta, ctx, p, my_rank, 1);
        }
        fi_cxi_ring_doorbell(meta, ctx);

        // Wait for signal from all peers
        for (uint32_t p = 0; p < num_peers; p++) {
            fi_cxi_signal_wait(meta, p, round);
        }
    }
    __syncthreads();
}
```

**Optimized barrier with tree reduction:**

For large peer counts, an all-to-all barrier is O(n²) in network traffic. A tree-based barrier reduces this:

```c
/* Tree barrier - logarithmic message complexity */
__global__ void tree_barrier(
    fi_ki_meta_t meta,
    uint32_t my_rank,
    uint32_t num_ranks,
    uint64_t round
) {
    if (threadIdx.x != 0) { __syncthreads(); return; }

    int ctx = 0;

    // Reduce phase: gather signals up the tree
    for (uint32_t stride = 1; stride < num_ranks; stride *= 2) {
        if ((my_rank % (stride * 2)) == 0) {
            uint32_t peer = my_rank + stride;
            if (peer < num_ranks) {
                fi_cxi_signal_wait(meta, peer, round);
            }
        } else if ((my_rank % stride) == 0) {
            uint32_t peer = my_rank - stride;
            fi_cxi_signal_send(meta, ctx, peer, my_rank, 1);
            fi_cxi_ring_doorbell(meta, ctx);
            break;  // Done contributing
        }
    }

    // Broadcast phase: propagate completion down the tree
    for (uint32_t stride = num_ranks / 2; stride >= 1; stride /= 2) {
        if ((my_rank % (stride * 2)) == 0) {
            uint32_t peer = my_rank + stride;
            if (peer < num_ranks) {
                fi_cxi_signal_send(meta, ctx, peer, my_rank, 1);
                fi_cxi_ring_doorbell(meta, ctx);
            }
        } else if ((my_rank % stride) == 0) {
            uint32_t peer = my_rank - stride;
            fi_cxi_signal_wait(meta, peer, round);
            break;
        }
    }

    __syncthreads();
}
```

### Example 4: Pipelined Send with Flow Control

Use signals for flow control when receiver has limited buffer space:

```c
__global__ void pipelined_send(
    fi_ki_meta_t meta,
    void const* src_buf,
    uint32_t num_chunks,
    uint32_t chunk_size,
    uint32_t peer,
    uint32_t window_size       /* Max in-flight chunks */
) {
    if (threadIdx.x != 0) return;

    int ctx = 0;
    uint64_t sent = 0;
    uint64_t acked = 0;

    while (sent < num_chunks) {
        // Wait for window space
        while (sent - acked >= window_size) {
            acked = fi_cxi_signal_read(meta, 0);
        }

        // Send chunk with signal to track completion
        void const* src = (char*)src_buf + sent * chunk_size;
        fi_cxi_inject_put(meta, ctx, peer, src, sent * chunk_size, chunk_size,
                          FI_KI_COOP_THREAD,
                          0, 1,                 /* increment peer's signal 0 */
                          FI_KI_NO_COUNTER);
        fi_cxi_ring_doorbell(meta, ctx);
        sent++;
    }

    // Wait for all acks
    fi_cxi_signal_wait(meta, 0, num_chunks);
}
```

---

## Memory Access Patterns

The metadata structure is designed to optimize GPU memory access efficiency.

### Peer Data: Structure-of-Arrays (SoA)

Peer routing data uses SoA layout for coalesced warp access:

```
Traditional AoS (bad):           SoA (good):
struct Peer {                    uint64_t dfa[N];      // contiguous
  uint64_t dfa;                  uint8_t  dfa_ext[N];  // contiguous
  uint8_t  dfa_ext;              uint32_t index_ext[N];// contiguous
  uint32_t index_ext;            uint64_t mr_base[N];  // contiguous
  uint64_t mr_base;              uint64_t mr_key[N];   // contiguous
  uint64_t mr_key;
};
Peer peers[N];  // scattered
```

**Access pattern when 32 warp threads load consecutive peer indices:**
| Field | Element Size | Warp Load | Cache Lines |
|-------|-------------|-----------|-------------|
| `dfa[base..base+31]` | 8 bytes | 256 bytes | 2 |
| `dfa_ext[base..base+31]` | 1 byte | 32 bytes | 1 (partial) |
| `index_ext[base..base+31]` | 4 bytes | 128 bytes | 1 |
| `mr_base[base..base+31]` | 8 bytes | 256 bytes | 2 |
| `mr_key[base..base+31]` | 8 bytes | 256 bytes | 2 |

Total: ~8 cache lines for all peer data (vs ~160 cache lines for AoS with 64-byte padding).

### Completion Data: All Contiguous

KI requires batch allocation (`alloc_counters_batch`, `alloc_target_cts_batch`)
to ensure all writeback buffers are contiguous. This enables direct array access:

**Counters and Target CTs:**
- Batch allocation places writebacks in caller-provided contiguous GPU buffer
- Direct array access: `cntr_wb[idx]`, `tgt_ct_wb[idx]`
- NIC DMAs to fixed offsets within the contiguous buffer
- 8 bytes per writeback (`struct c_ct_writeback`)

**Signals:**
- Contiguous array in GPU memory
- Remote peers FAA into this array
- Direct array access: `signals[idx]`
- 8 bytes per signal (`uint64_t`)

### Cache Line Isolation

Write-heavy data (command queue write pointers) is isolated to separate cache lines
to prevent false sharing between CTAs:

```c
struct fi_cxi_ki_cmdq_hot {
    volatile uint64_t *wp_addr;
    volatile uint32_t  wp;
    uint32_t           _pad[13];  /* Pad to 128 bytes (GPU L2 cache line) */
} __attribute__((aligned(128)));
```

---

## Performance Guidelines

### Context Usage

Using multiple command contexts can improve performance by reducing contention:

```c
/* Assign contexts to CTAs for parallelism */
int ctx = blockIdx.x % meta->context_count;
fi_cxi_inject_put_simple(meta, ctx, peer, ...);
```

For best performance:
- Use one context per CTA when possible
- Avoid sharing contexts across warps within a CTA if contention is high

### Signal Allocation

For performance-oriented kernels, keep signals exclusive to each CTA:

```c
/* Each CTA uses its own signal index */
uint32_t my_signal = blockIdx.x;
fi_cxi_signal_wait(meta, my_signal, expected);
```

Sharing signals across CTAs requires additional synchronization overhead.

### Batching Commands

Ring the doorbell after batching multiple commands, not after each one:

```c
/* Post all commands first */
for (int i = 0; i < num_ops; i++) {
    fi_cxi_inject_put_simple(meta, ctx, peers[i], ...);
}
/* Then ring once */
fi_cxi_ring_doorbell(meta, ctx);
```

---

## API Summary

| Function | Model | Purpose |
|----------|-------|---------|
| `fi_cxi_inject_put` | B | PUT with initiator signal |
| `fi_cxi_inject_put_tagged` | A | PUT with match bits for CT routing |
| `fi_cxi_inject_put_simple` | A | PUT without match bits |
| `fi_cxi_ring_doorbell` | - | Notify NIC of commands |
| `fi_cxi_flush` | - | Wait for local consumption |
| `fi_cxi_cntr_read` | - | Read local counter |
| `fi_cxi_cntr_wait` | - | Wait for local counter |
| `fi_cxi_cntr_reset` | - | Reset local counter |
| `fi_cxi_target_ct_read` | A | Read target CT |
| `fi_cxi_target_ct_wait` | A | Wait for target CT |
| `fi_cxi_target_ct_reset` | A | Reset target CT |
| `fi_cxi_signal_read` | B | Read signal |
| `fi_cxi_signal_wait` | B | Wait for signal |
| `fi_cxi_signal_reset` | B | Reset signal |
| `fi_cxi_signal_send` | B | Send signal (no data) |
