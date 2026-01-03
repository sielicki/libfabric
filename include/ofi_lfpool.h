/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

/*
 * ofi_lfpool.h - Lock-free buffer pool for libfabric
 *
 * A bounded, lock-free buffer pool using Vyukov's MPMC queue algorithm.
 * Provides O(1) allocation and deallocation without locks, suitable for
 * high-performance multi-threaded buffer management.
 *
 * The pool is conceptually a queue that starts full:
 * - alloc() = dequeue (consume a buffer)
 * - free() = enqueue (return a buffer)
 *
 * Design:
 * - Fixed-size pool of entries, allocated contiguously for cache efficiency
 * - Lock-free using Vyukov's bounded MPMC queue algorithm
 * - Each slot has a sequence number to prevent ABA problems
 * - Separate cache lines for read/write positions to avoid false sharing
 *
 * The sequence number protocol:
 * - slot.seq == pos means slot is ready for dequeue at dequeue_pos == pos
 * - slot.seq == pos + capacity means slot is ready for enqueue at enqueue_pos == pos
 *
 * Initially all slots are ready for dequeue (seq[i] = i), and dequeue_pos = 0.
 *
 * Usage:
 *   struct ofi_lfpool pool;
 *   ofi_lfpool_init(&pool, 1024, sizeof(my_struct), 64);
 *
 *   void *ptr = ofi_lfpool_alloc(&pool);
 *   if (ptr) {
 *       // use ptr
 *       ofi_lfpool_free(&pool, ptr);
 *   }
 *
 *   ofi_lfpool_fini(&pool);
 */

#ifndef _OFI_LFPOOL_H_
#define _OFI_LFPOOL_H_

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <rdma/fi_errno.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OFI_LFPOOL_CACHE_LINE 64

/*
 * Synchronization mode flags for producer/consumer optimization.
 * These can be combined: e.g., OFI_LFPOOL_SP | OFI_LFPOOL_MC
 */
enum ofi_lfpool_sync_mode {
	OFI_LFPOOL_MP = 0,      /* Multi-producer (default) */
	OFI_LFPOOL_SP = 1 << 0, /* Single-producer (free) */
	OFI_LFPOOL_MC = 0,      /* Multi-consumer (default) */
	OFI_LFPOOL_SC = 1 << 1, /* Single-consumer (alloc) */
};

/*
 * Freelist slot - contains sequence number and pointer to buffer.
 * Aligned to cache line for optimal atomic access.
 */
struct ofi_lfpool_slot {
	_Atomic int64_t seq;
	void *ptr;
	char pad[OFI_LFPOOL_CACHE_LINE - sizeof(_Atomic int64_t) - sizeof(void *)];
} __attribute__((aligned(OFI_LFPOOL_CACHE_LINE)));

/*
 * Lock-free buffer pool structure.
 *
 * Memory layout ensures no false sharing:
 * - dequeue_pos (alloc) on its own cache line
 * - enqueue_pos (free) on its own cache line
 * - metadata on its own cache line
 * - slots array follows
 */
struct ofi_lfpool {
	/* Consumer (alloc) position - separate cache line */
	_Atomic int64_t dequeue_pos;
	char pad0[OFI_LFPOOL_CACHE_LINE - sizeof(_Atomic int64_t)];

	/* Producer (free) position - separate cache line */
	_Atomic int64_t enqueue_pos;
	char pad1[OFI_LFPOOL_CACHE_LINE - sizeof(_Atomic int64_t)];

	/* Metadata - separate cache line */
	size_t capacity;       /* Number of slots (power of 2) */
	size_t capacity_mask;  /* capacity - 1, for fast modulo */
	size_t entry_size;     /* Size of each buffer */
	void *buffer_base;     /* Base of contiguous buffer region */
	struct ofi_lfpool_slot *slots; /* Freelist slots */
	unsigned int sync_mode;/* SP/SC/MP/MC flags */
	char pad2[OFI_LFPOOL_CACHE_LINE -
		  ((5 * sizeof(void *)) + sizeof(unsigned int)) %
		  OFI_LFPOOL_CACHE_LINE];
};

/*
 * Round up to next power of 2.
 */
static inline size_t ofi_lfpool_next_pow2(size_t n)
{
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n |= n >> 32;
	return n + 1;
}

/*
 * ofi_lfpool_init_ex() - Initialize a lock-free buffer pool with mode flags.
 *
 * @pool:       Pool structure to initialize
 * @capacity:   Number of buffers (will be rounded up to power of 2)
 * @entry_size: Size of each buffer in bytes
 * @alignment:  Alignment requirement for each buffer (0 for default)
 * @sync_mode:  Synchronization mode flags (OFI_LFPOOL_SP, OFI_LFPOOL_SC, etc.)
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 */
static inline int ofi_lfpool_init_ex(struct ofi_lfpool *pool,
				     size_t capacity,
				     size_t entry_size,
				     size_t alignment,
				     unsigned int sync_mode)
{
	size_t i;
	size_t aligned_size;
	char *buf;

	if (capacity == 0)
		capacity = 1;

	/* Round capacity to power of 2 for fast modulo */
	capacity = ofi_lfpool_next_pow2(capacity);

	/* Default alignment to cache line */
	if (alignment == 0)
		alignment = OFI_LFPOOL_CACHE_LINE;

	/* Round entry size up to alignment */
	aligned_size = (entry_size + alignment - 1) & ~(alignment - 1);

	/* Allocate slot array */
	pool->slots = aligned_alloc(OFI_LFPOOL_CACHE_LINE,
				    capacity * sizeof(struct ofi_lfpool_slot));
	if (!pool->slots)
		return -FI_ENOMEM;

	/* Allocate contiguous buffer region */
	pool->buffer_base = aligned_alloc(alignment, capacity * aligned_size);
	if (!pool->buffer_base) {
		free(pool->slots);
		pool->slots = NULL;
		return -FI_ENOMEM;
	}

	pool->capacity = capacity;
	pool->capacity_mask = capacity - 1;
	pool->entry_size = aligned_size;
	pool->sync_mode = sync_mode;

	/*
	 * Initialize as a full queue:
	 * - dequeue_pos = 0 (first slot to dequeue)
	 * - enqueue_pos = capacity (next slot to enqueue, which wraps to 0)
	 *
	 * Actually, for Vyukov's algorithm with a full queue:
	 * - All slots have seq = index (ready for dequeue at pos = index)
	 * - dequeue_pos = 0
	 * - enqueue_pos = 0 (but all slots are "full" so enqueue will fail until dequeue)
	 *
	 * The key: seq[i] = i means slot i is ready for dequeue at dequeue_pos = i.
	 * After dequeue at pos P, we set seq = P + capacity.
	 * For enqueue at pos P, we wait for seq == P + capacity.
	 *
	 * With enqueue_pos = 0 initially:
	 * - enqueue looks at slot[0], expects seq = 0 + capacity = capacity
	 * - But seq[0] = 0, so diff = 0 - capacity = -capacity < 0
	 * - This means "queue is full", which is correct!
	 */
	atomic_store_explicit(&pool->dequeue_pos, 0, memory_order_relaxed);
	atomic_store_explicit(&pool->enqueue_pos, 0, memory_order_relaxed);

	/* Initialize all slots as "full" (ready for dequeue).
	 *
	 * In Vyukov's algorithm:
	 * - Producer sets seq = pos + 1 after producing
	 * - Consumer expects seq == pos + 1 before consuming
	 *
	 * For a pool starting full (all slots ready for alloc/consume):
	 * - seq[i] = i + 1 means ready for dequeue at dequeue_pos = i
	 */
	buf = pool->buffer_base;
	for (i = 0; i < capacity; i++) {
		atomic_store_explicit(&pool->slots[i].seq, (int64_t)(i + 1),
				      memory_order_relaxed);
		pool->slots[i].ptr = buf + (i * aligned_size);
	}

	/* Memory fence to ensure all initialization is visible */
	atomic_thread_fence(memory_order_release);

	return 0;
}

/*
 * ofi_lfpool_init() - Initialize a lock-free buffer pool (MPMC mode).
 *
 * Convenience wrapper that initializes with default MPMC mode.
 */
static inline int ofi_lfpool_init(struct ofi_lfpool *pool,
				  size_t capacity,
				  size_t entry_size,
				  size_t alignment)
{
	return ofi_lfpool_init_ex(pool, capacity, entry_size, alignment,
				  OFI_LFPOOL_MP | OFI_LFPOOL_MC);
}

/*
 * ofi_lfpool_fini() - Clean up and free pool resources.
 */
static inline void ofi_lfpool_fini(struct ofi_lfpool *pool)
{
	free(pool->buffer_base);
	free(pool->slots);
	pool->buffer_base = NULL;
	pool->slots = NULL;
	pool->capacity = 0;
}

/*
 * ofi_lfpool_alloc_sc() - Single-consumer fast path alloc.
 *
 * Optimized for single-consumer mode - no CAS needed for position update.
 */
static inline void *ofi_lfpool_alloc_sc(struct ofi_lfpool *pool)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq;
	void *ptr;

	pos = atomic_load_explicit(&pool->dequeue_pos, memory_order_relaxed);
	slot = &pool->slots[pos & pool->capacity_mask];
	seq = atomic_load_explicit(&slot->seq, memory_order_acquire);

	if (seq != pos + 1)
		return NULL; /* Queue empty */

	ptr = slot->ptr;

	/* No CAS needed - we're the only consumer */
	atomic_store_explicit(&pool->dequeue_pos, pos + 1,
			      memory_order_relaxed);
	atomic_store_explicit(&slot->seq, pos + (int64_t)pool->capacity,
			      memory_order_release);
	return ptr;
}

/*
 * ofi_lfpool_alloc_mc() - Multi-consumer alloc (Vyukov algorithm).
 *
 * Full lock-free algorithm with CAS for multi-consumer safety.
 */
static inline void *ofi_lfpool_alloc_mc(struct ofi_lfpool *pool)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq, diff;
	void *ptr;

	pos = atomic_load_explicit(&pool->dequeue_pos, memory_order_relaxed);

	for (;;) {
		slot = &pool->slots[pos & pool->capacity_mask];
		seq = atomic_load_explicit(&slot->seq, memory_order_acquire);

		/* Vyukov consumer: expects seq == pos + 1
		 * diff = seq - (pos + 1)
		 *   == 0: slot is ready for dequeue
		 *   < 0:  slot not yet enqueued (queue empty)
		 *   > 0:  slot already dequeued by another thread
		 */
		diff = seq - (pos + 1);

		if (diff == 0) {
			/* Slot is ready - try to claim it */
			if (atomic_compare_exchange_weak_explicit(
				    &pool->dequeue_pos, &pos, pos + 1,
				    memory_order_relaxed,
				    memory_order_relaxed)) {
				/* Successfully claimed */
				ptr = slot->ptr;

				/*
				 * Mark slot as ready for enqueue.
				 * Set seq = pos + capacity so that enqueue_pos = pos
				 * can use this slot (it will see diff = seq - (pos + cap) = 0).
				 */
				atomic_store_explicit(&slot->seq,
						      pos + (int64_t)pool->capacity,
						      memory_order_release);
				return ptr;
			}
			/* CAS failed, pos was updated by another thread */
		} else if (diff < 0) {
			/* Queue is empty at this position */
			return NULL;
		} else {
			/* Slot already dequeued, reload position */
			pos = atomic_load_explicit(&pool->dequeue_pos,
						   memory_order_relaxed);
		}
	}
}

/*
 * ofi_lfpool_alloc() - Allocate a buffer from the pool.
 *
 * Lock-free dequeue operation. Takes a buffer from the "full" side of the queue.
 * Dispatches to SC or MC variant based on pool mode.
 *
 * @pool: Buffer pool
 *
 * Returns: Pointer to allocated buffer, or NULL if pool is empty
 */
static inline void *ofi_lfpool_alloc(struct ofi_lfpool *pool)
{
	if (pool->sync_mode & OFI_LFPOOL_SC)
		return ofi_lfpool_alloc_sc(pool);
	return ofi_lfpool_alloc_mc(pool);
}

/*
 * ofi_lfpool_alloc_spin() - Allocate a buffer, spinning until available.
 *
 * Spins until a buffer becomes available. Use when you know a free is
 * imminent (e.g., bounded producer-consumer). Includes exponential backoff
 * with pause instructions to reduce contention.
 *
 * @pool: Buffer pool
 *
 * Returns: Pointer to allocated buffer (never NULL)
 */
static inline void *ofi_lfpool_alloc_spin(struct ofi_lfpool *pool)
{
	void *ptr;
	unsigned int backoff = 1;

	while ((ptr = ofi_lfpool_alloc(pool)) == NULL) {
		/* Exponential backoff with pause hints */
		for (unsigned int i = 0; i < backoff; i++) {
#if defined(__x86_64__) || defined(__i386__)
			__asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__)
			__asm__ __volatile__("yield" ::: "memory");
#else
			atomic_thread_fence(memory_order_seq_cst);
#endif
		}
		if (backoff < 1024)
			backoff *= 2;
	}
	return ptr;
}

/*
 * ofi_lfpool_free_sp() - Single-producer fast path free.
 *
 * Optimized for single-producer mode - no CAS needed for position update.
 */
static inline int ofi_lfpool_free_sp(struct ofi_lfpool *pool, void *ptr)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq;

	pos = atomic_load_explicit(&pool->enqueue_pos, memory_order_relaxed);
	slot = &pool->slots[pos & pool->capacity_mask];
	seq = atomic_load_explicit(&slot->seq, memory_order_acquire);

	if (seq != pos + (int64_t)pool->capacity)
		return -FI_EAGAIN; /* Queue full */

	slot->ptr = ptr;

	/* No CAS needed - we're the only producer */
	atomic_store_explicit(&pool->enqueue_pos, pos + 1,
			      memory_order_relaxed);
	atomic_store_explicit(&slot->seq, pos + (int64_t)pool->capacity + 1,
			      memory_order_release);
	return 0;
}

/*
 * ofi_lfpool_free_mp() - Multi-producer free (Vyukov algorithm).
 *
 * Full lock-free algorithm with CAS for multi-producer safety.
 */
static inline int ofi_lfpool_free_mp(struct ofi_lfpool *pool, void *ptr)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq, diff;

	pos = atomic_load_explicit(&pool->enqueue_pos, memory_order_relaxed);

	for (;;) {
		slot = &pool->slots[pos & pool->capacity_mask];
		seq = atomic_load_explicit(&slot->seq, memory_order_acquire);

		/*
		 * For enqueue at position P, we need slot to be "consumed"
		 * (dequeued). After dequeue at dequeue_pos = P, slot has
		 * seq = P + capacity.
		 *
		 * diff = seq - (pos + capacity)
		 *   == 0: slot was dequeued, ready for enqueue
		 *   < 0:  slot not yet dequeued (still has seq = old dequeue position)
		 *   > 0:  slot already enqueued by another thread
		 */
		diff = seq - (pos + (int64_t)pool->capacity);

		if (diff == 0) {
			/* Slot is ready - try to claim it */
			if (atomic_compare_exchange_weak_explicit(
				    &pool->enqueue_pos, &pos, pos + 1,
				    memory_order_relaxed,
				    memory_order_relaxed)) {
				/* Successfully claimed */
				slot->ptr = ptr;

				/*
				 * Mark slot as ready for dequeue.
				 *
				 * Vyukov producer sets seq = pos + 1.
				 * Next consumer at this slot: dequeue_pos = pos + capacity.
				 * Consumer expects seq == (pos + capacity) + 1.
				 * So set seq = pos + capacity + 1.
				 */
				atomic_store_explicit(&slot->seq,
						      pos + (int64_t)pool->capacity + 1,
						      memory_order_release);
				return 0;
			}
			/* CAS failed, pos was updated by another thread */
		} else if (diff < 0) {
			/* Queue is full - slot not yet dequeued */
			return -FI_EAGAIN;
		} else {
			/* Slot already enqueued, reload position */
			pos = atomic_load_explicit(&pool->enqueue_pos,
						   memory_order_relaxed);
		}
	}
}

/*
 * ofi_lfpool_free() - Return a buffer to the pool.
 *
 * Lock-free enqueue operation. Returns a buffer to the "empty" side of the queue.
 * Dispatches to SP or MP variant based on pool mode.
 *
 * @pool: Buffer pool
 * @ptr:  Pointer previously returned by ofi_lfpool_alloc()
 *
 * Returns: 0 on success, -FI_EAGAIN if queue is full (should never happen
 *          if you only free what you allocated)
 */
static inline int ofi_lfpool_free(struct ofi_lfpool *pool, void *ptr)
{
	if (pool->sync_mode & OFI_LFPOOL_SP)
		return ofi_lfpool_free_sp(pool, ptr);
	return ofi_lfpool_free_mp(pool, ptr);
}

/*
 * ofi_lfpool_alloc_bulk() - Allocate multiple buffers from the pool.
 *
 * Reserves n slots atomically, then fills the provided array.
 * Much more efficient than n individual alloc() calls when n > 1.
 *
 * @pool:   Buffer pool
 * @ptrs:   Array to fill with allocated pointers
 * @n:      Number of buffers to allocate
 *
 * Returns: Number of buffers actually allocated (may be less than n if pool
 *          doesn't have enough available)
 */
static inline size_t ofi_lfpool_alloc_bulk(struct ofi_lfpool *pool,
					   void **ptrs, size_t n)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq, diff;
	size_t i, allocated;

	if (n == 0)
		return 0;

	pos = atomic_load_explicit(&pool->dequeue_pos, memory_order_relaxed);

	for (;;) {
		/* Check how many slots are available starting at pos */
		allocated = 0;
		for (i = 0; i < n; i++) {
			slot = &pool->slots[(pos + i) & pool->capacity_mask];
			seq = atomic_load_explicit(&slot->seq,
						   memory_order_acquire);
			diff = seq - (pos + (int64_t)i + 1);
			if (diff != 0)
				break;
			allocated++;
		}

		if (allocated == 0)
			return 0; /* Pool empty */

		/* Try to reserve 'allocated' slots */
		if (pool->sync_mode & OFI_LFPOOL_SC) {
			/* Single-consumer: no CAS needed */
			atomic_store_explicit(&pool->dequeue_pos,
					      pos + (int64_t)allocated,
					      memory_order_relaxed);
			break;
		} else {
			/* Multi-consumer: use CAS */
			if (atomic_compare_exchange_weak_explicit(
				    &pool->dequeue_pos, &pos,
				    pos + (int64_t)allocated,
				    memory_order_relaxed,
				    memory_order_relaxed))
				break;
			/* CAS failed, retry */
		}
	}

	/* Extract pointers and mark slots as ready for enqueue */
	for (i = 0; i < allocated; i++) {
		slot = &pool->slots[(pos + i) & pool->capacity_mask];
		ptrs[i] = slot->ptr;
		atomic_store_explicit(&slot->seq,
				      pos + (int64_t)i + (int64_t)pool->capacity,
				      memory_order_release);
	}

	return allocated;
}

/*
 * ofi_lfpool_free_bulk() - Return multiple buffers to the pool.
 *
 * Reserves n slots atomically, then fills them with the provided pointers.
 * Much more efficient than n individual free() calls when n > 1.
 *
 * @pool:   Buffer pool
 * @ptrs:   Array of pointers to free
 * @n:      Number of buffers to free
 *
 * Returns: Number of buffers actually freed (may be less than n if pool
 *          is nearly full - should never happen in correct usage)
 */
static inline size_t ofi_lfpool_free_bulk(struct ofi_lfpool *pool,
					  void **ptrs, size_t n)
{
	struct ofi_lfpool_slot *slot;
	int64_t pos, seq, diff;
	size_t i, freed;

	if (n == 0)
		return 0;

	pos = atomic_load_explicit(&pool->enqueue_pos, memory_order_relaxed);

	for (;;) {
		/* Check how many slots are available starting at pos */
		freed = 0;
		for (i = 0; i < n; i++) {
			slot = &pool->slots[(pos + i) & pool->capacity_mask];
			seq = atomic_load_explicit(&slot->seq,
						   memory_order_acquire);
			diff = seq - (pos + (int64_t)i + (int64_t)pool->capacity);
			if (diff != 0)
				break;
			freed++;
		}

		if (freed == 0)
			return 0; /* Pool full */

		/* Try to reserve 'freed' slots */
		if (pool->sync_mode & OFI_LFPOOL_SP) {
			/* Single-producer: no CAS needed */
			atomic_store_explicit(&pool->enqueue_pos,
					      pos + (int64_t)freed,
					      memory_order_relaxed);
			break;
		} else {
			/* Multi-producer: use CAS */
			if (atomic_compare_exchange_weak_explicit(
				    &pool->enqueue_pos, &pos,
				    pos + (int64_t)freed,
				    memory_order_relaxed,
				    memory_order_relaxed))
				break;
			/* CAS failed, retry */
		}
	}

	/* Store pointers and mark slots as ready for dequeue */
	for (i = 0; i < freed; i++) {
		slot = &pool->slots[(pos + i) & pool->capacity_mask];
		slot->ptr = ptrs[i];
		atomic_store_explicit(&slot->seq,
				      pos + (int64_t)i + (int64_t)pool->capacity + 1,
				      memory_order_release);
	}

	return freed;
}

/*
 * ofi_lfpool_capacity() - Return the pool capacity.
 */
static inline size_t ofi_lfpool_capacity(struct ofi_lfpool *pool)
{
	return pool->capacity;
}

/*
 * ofi_lfpool_entry_size() - Return the aligned entry size.
 */
static inline size_t ofi_lfpool_entry_size(struct ofi_lfpool *pool)
{
	return pool->entry_size;
}

#ifdef __cplusplus
}
#endif

#endif /* _OFI_LFPOOL_H_ */
