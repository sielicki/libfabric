/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2024 Hewlett Packard Enterprise Development LP
 */

/*
 * cxip/lfqueue.h - Lock-free data structures for CXI provider
 *
 * This header provides lock-free alternatives to the standard libfabric
 * data structures used in the CXI provider's critical path.
 *
 * Components:
 * - cxip_req_id_pool: Lock-free 16-bit request ID allocation
 *   Replaces: struct indexer + ofi_idx_insert/remove
 *
 * Usage model:
 * - Request IDs are allocated from a pool using atomic CAS
 * - Each slot in the pool holds a pointer to the request
 * - Lookup is O(1) direct array access
 * - Allocation/free are lock-free
 */

#ifndef _CXIP_LFQUEUE_H_
#define _CXIP_LFQUEUE_H_

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <rdma/fi_errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lock-free request ID pool.
 *
 * Provides O(1) allocation and lookup of 16-bit request IDs.
 * Each ID maps to a request pointer that can be atomically set/cleared.
 *
 * Design:
 * - Fixed array of atomic pointers indexed by request ID
 * - Allocation uses atomic increment of next_id + CAS to claim slot
 * - Free uses atomic store of NULL
 * - Lookup is direct array access (no CAS needed for read)
 *
 * The pool uses ID 0 as invalid (never allocated), so effective
 * capacity is (capacity - 1).
 */

#define CXIP_REQ_ID_INVALID 0

struct cxip_req_id_pool {
	_Atomic uint32_t next_id;      /* Next ID to try allocating */
	_Atomic uint32_t alloc_count;  /* Number of allocated IDs (debug) */
	uint32_t capacity;             /* Total slots (power of 2) */
	uint32_t capacity_mask;        /* capacity - 1 for fast modulo */
	_Atomic uintptr_t slots[];     /* Flexible array of request pointers */
};

/*
 * Round up to next power of 2.
 */
static inline uint32_t cxip_req_id_next_pow2(uint32_t n)
{
	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	return n + 1;
}

/*
 * cxip_req_id_pool_create() - Create a lock-free request ID pool.
 *
 * @capacity:  Maximum number of concurrent request IDs (rounded to power of 2)
 * @pool_out:  Output pointer to created pool
 *
 * Returns: 0 on success, -FI_ENOMEM on allocation failure
 */
static inline int cxip_req_id_pool_create(uint32_t capacity,
					  struct cxip_req_id_pool **pool_out)
{
	struct cxip_req_id_pool *pool;
	size_t alloc_size;

	if (capacity < 2)
		capacity = 2;

	capacity = cxip_req_id_next_pow2(capacity);

	alloc_size = sizeof(*pool) + capacity * sizeof(_Atomic uintptr_t);
	pool = calloc(1, alloc_size);
	if (!pool)
		return -FI_ENOMEM;

	pool->capacity = capacity;
	pool->capacity_mask = capacity - 1;
	atomic_store_explicit(&pool->next_id, 1, memory_order_relaxed);
	atomic_store_explicit(&pool->alloc_count, 0, memory_order_relaxed);

	/* All slots start as NULL (available) */
	for (uint32_t i = 0; i < capacity; i++)
		atomic_store_explicit(&pool->slots[i], 0, memory_order_relaxed);

	atomic_thread_fence(memory_order_release);

	*pool_out = pool;
	return 0;
}

/*
 * cxip_req_id_pool_destroy() - Destroy a request ID pool.
 */
static inline void cxip_req_id_pool_destroy(struct cxip_req_id_pool *pool)
{
	free(pool);
}

/*
 * cxip_req_id_alloc() - Allocate a request ID and associate it with a request.
 *
 * @pool:  Request ID pool
 * @req:   Request pointer to associate with the ID (must not be NULL)
 *
 * Returns: Allocated ID (1 to capacity-1), or -FI_EAGAIN if pool is full
 *
 * This function is lock-free and safe to call from multiple threads.
 */
static inline int cxip_req_id_alloc(struct cxip_req_id_pool *pool, void *req)
{
	uint32_t id, start_id;
	uintptr_t expected;
	uint32_t attempts = 0;

	if (!req)
		return -FI_EINVAL;

	start_id = atomic_fetch_add_explicit(&pool->next_id, 1,
					     memory_order_relaxed);

	/* Try slots starting from start_id, wrapping around */
	for (attempts = 0; attempts < pool->capacity; attempts++) {
		id = (start_id + attempts) & pool->capacity_mask;

		/* Skip ID 0 (reserved as invalid) */
		if (id == 0)
			continue;

		expected = 0;
		if (atomic_compare_exchange_weak_explicit(
			    &pool->slots[id], &expected, (uintptr_t)req,
			    memory_order_acq_rel, memory_order_relaxed)) {
			atomic_fetch_add_explicit(&pool->alloc_count, 1,
						  memory_order_relaxed);
			return (int)id;
		}
	}

	return -FI_EAGAIN;
}

/*
 * cxip_req_id_free() - Free a request ID.
 *
 * @pool:  Request ID pool
 * @id:    ID to free (must have been returned by cxip_req_id_alloc)
 *
 * Returns: The request pointer that was associated with the ID,
 *          or NULL if the ID was invalid or already freed.
 *
 * This function is lock-free and safe to call from multiple threads.
 */
static inline void *cxip_req_id_free(struct cxip_req_id_pool *pool, int id)
{
	uintptr_t old_ptr;

	if (id <= 0 || (uint32_t)id >= pool->capacity)
		return NULL;

	old_ptr = atomic_exchange_explicit(&pool->slots[id], 0,
					   memory_order_acq_rel);

	if (old_ptr) {
		atomic_fetch_sub_explicit(&pool->alloc_count, 1,
					  memory_order_relaxed);
	}

	return (void *)old_ptr;
}

/*
 * cxip_req_id_lookup() - Look up a request by ID.
 *
 * @pool:  Request ID pool
 * @id:    ID to look up
 *
 * Returns: Request pointer, or NULL if ID is invalid or not allocated.
 *
 * This function is lock-free. The returned pointer is valid as long as
 * the caller ensures the request is not freed concurrently.
 */
static inline void *cxip_req_id_lookup(struct cxip_req_id_pool *pool, int id)
{
	if (id <= 0 || (uint32_t)id >= pool->capacity)
		return NULL;

	return (void *)atomic_load_explicit(&pool->slots[id],
					    memory_order_acquire);
}

/*
 * cxip_req_id_count() - Get the number of allocated IDs (approximate).
 */
static inline uint32_t cxip_req_id_count(struct cxip_req_id_pool *pool)
{
	return atomic_load_explicit(&pool->alloc_count, memory_order_relaxed);
}

#ifdef __cplusplus
}
#endif

#endif /* _CXIP_LFQUEUE_H_ */
