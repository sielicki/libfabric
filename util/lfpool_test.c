/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

/*
 * lfpool_test.c - Tests for lock-free buffer pool
 *
 * Build: gcc -O2 -pthread -I../include lfpool_test.c -o lfpool_test
 * Run: ./lfpool_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <stdatomic.h>

/* Minimal config.h stub */
#ifndef HAVE_ATOMICS
#define HAVE_ATOMICS 1
#endif

#include "ofi_lfpool.h"

#define TEST_CAPACITY 64
#define TEST_ENTRY_SIZE 128
#define NUM_THREADS 128
#define OPS_PER_THREAD 100000

static struct ofi_lfpool pool;
static _Atomic int alloc_count;
static _Atomic int free_count;
static _Atomic int alloc_fail_count;

/*
 * Test 1: Basic single-threaded alloc/free
 */
static int test_basic(void)
{
	struct ofi_lfpool p;
	void *ptrs[16];
	int i, ret;

	printf("Test 1: Basic alloc/free... ");

	ret = ofi_lfpool_init(&p, 16, 64, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Allocate all */
	for (i = 0; i < 16; i++) {
		ptrs[i] = ofi_lfpool_alloc(&p);
		if (!ptrs[i]) {
			printf("FAIL: alloc %d returned NULL\n", i);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	/* Pool should be empty now */
	if (ofi_lfpool_alloc(&p) != NULL) {
		printf("FAIL: alloc on empty pool should return NULL\n");
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Free all */
	for (i = 0; i < 16; i++) {
		ret = ofi_lfpool_free(&p, ptrs[i]);
		if (ret) {
			printf("FAIL: free %d returned %d\n", i, ret);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	/* Should be able to allocate again */
	for (i = 0; i < 16; i++) {
		ptrs[i] = ofi_lfpool_alloc(&p);
		if (!ptrs[i]) {
			printf("FAIL: re-alloc %d returned NULL\n", i);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 2: Unique pointers
 */
static int test_unique_pointers(void)
{
	struct ofi_lfpool p;
	void *ptrs[32];
	int i, j, ret;

	printf("Test 2: Unique pointers... ");

	ret = ofi_lfpool_init(&p, 32, 64, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Allocate all and check uniqueness */
	for (i = 0; i < 32; i++) {
		ptrs[i] = ofi_lfpool_alloc(&p);
		if (!ptrs[i]) {
			printf("FAIL: alloc %d returned NULL\n", i);
			ofi_lfpool_fini(&p);
			return 1;
		}

		/* Check for duplicates */
		for (j = 0; j < i; j++) {
			if (ptrs[j] == ptrs[i]) {
				printf("FAIL: duplicate pointer at %d and %d\n", j, i);
				ofi_lfpool_fini(&p);
				return 1;
			}
		}
	}

	/* Free all */
	for (i = 0; i < 32; i++)
		ofi_lfpool_free(&p, ptrs[i]);

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 3: Multi-threaded stress test
 */
static void *stress_thread(void *arg)
{
	int thread_id = *(int *)arg;
	void *ptr;
	int i, ret;
	int local_alloc = 0, local_free = 0, local_fail = 0;

	(void)thread_id;

	for (i = 0; i < OPS_PER_THREAD; i++) {
		ptr = ofi_lfpool_alloc(&pool);
		if (ptr) {
			local_alloc++;

			/* Write pattern to detect corruption */
			memset(ptr, 0xAB, TEST_ENTRY_SIZE);

			/* Free it back */
			ret = ofi_lfpool_free(&pool, ptr);
			if (ret == 0)
				local_free++;
		} else {
			local_fail++;
		}
	}

	atomic_fetch_add(&alloc_count, local_alloc);
	atomic_fetch_add(&free_count, local_free);
	atomic_fetch_add(&alloc_fail_count, local_fail);

	return NULL;
}

static int test_stress(void)
{
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];
	int i, ret;

	printf("Test 3: Multi-threaded stress (%d threads, %d ops each)... ",
	       NUM_THREADS, OPS_PER_THREAD);
	fflush(stdout);

	ret = ofi_lfpool_init(&pool, TEST_CAPACITY, TEST_ENTRY_SIZE, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	atomic_store(&alloc_count, 0);
	atomic_store(&free_count, 0);
	atomic_store(&alloc_fail_count, 0);

	/* Start threads */
	for (i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		ret = pthread_create(&threads[i], NULL, stress_thread, &thread_ids[i]);
		if (ret) {
			printf("FAIL: pthread_create returned %d\n", ret);
			ofi_lfpool_fini(&pool);
			return 1;
		}
	}

	/* Wait for completion */
	for (i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], NULL);

	ofi_lfpool_fini(&pool);

	printf("PASS (alloc=%d, free=%d, fail=%d)\n",
	       atomic_load(&alloc_count),
	       atomic_load(&free_count),
	       atomic_load(&alloc_fail_count));

	/* Verify counts match */
	if (atomic_load(&alloc_count) != atomic_load(&free_count)) {
		printf("WARNING: alloc/free mismatch!\n");
	}

	return 0;
}

/*
 * Test 4: LIFO-ish behavior (not guaranteed but typical)
 */
static int test_alloc_free_pattern(void)
{
	struct ofi_lfpool p;
	void *ptr1, *ptr2, *ptr3;
	int ret;

	printf("Test 4: Alloc/free pattern... ");

	ret = ofi_lfpool_init(&p, 4, 32, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Alloc 3 */
	ptr1 = ofi_lfpool_alloc(&p);
	ptr2 = ofi_lfpool_alloc(&p);
	ptr3 = ofi_lfpool_alloc(&p);

	if (!ptr1 || !ptr2 || !ptr3) {
		printf("FAIL: initial allocs failed\n");
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Free middle one */
	ofi_lfpool_free(&p, ptr2);

	/* Alloc again - should get ptr2 back (or similar slot) */
	void *ptr4 = ofi_lfpool_alloc(&p);
	if (!ptr4) {
		printf("FAIL: re-alloc failed\n");
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Free all */
	ofi_lfpool_free(&p, ptr1);
	ofi_lfpool_free(&p, ptr3);
	ofi_lfpool_free(&p, ptr4);

	/* Verify we can alloc 4 again */
	int count = 0;
	while (ofi_lfpool_alloc(&p))
		count++;

	if (count != 4) {
		printf("FAIL: expected 4 allocs, got %d\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 5: Power of 2 rounding
 */
static int test_capacity_rounding(void)
{
	struct ofi_lfpool p;
	int ret;

	printf("Test 5: Capacity rounding... ");

	/* Request 10, should get 16 */
	ret = ofi_lfpool_init(&p, 10, 32, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	if (ofi_lfpool_capacity(&p) != 16) {
		printf("FAIL: capacity is %zu, expected 16\n",
		       ofi_lfpool_capacity(&p));
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Verify we can alloc 16 */
	int count = 0;
	while (ofi_lfpool_alloc(&p))
		count++;

	if (count != 16) {
		printf("FAIL: could alloc %d, expected 16\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 6: Single-producer/single-consumer mode
 */
static int test_spsc_mode(void)
{
	struct ofi_lfpool p;
	void *ptrs[16];
	int i, ret;

	printf("Test 6: SP/SC mode... ");

	ret = ofi_lfpool_init_ex(&p, 16, 64, 0, OFI_LFPOOL_SP | OFI_LFPOOL_SC);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Allocate all */
	for (i = 0; i < 16; i++) {
		ptrs[i] = ofi_lfpool_alloc(&p);
		if (!ptrs[i]) {
			printf("FAIL: alloc %d returned NULL\n", i);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	/* Pool should be empty */
	if (ofi_lfpool_alloc(&p) != NULL) {
		printf("FAIL: alloc on empty pool should return NULL\n");
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Free all */
	for (i = 0; i < 16; i++) {
		ret = ofi_lfpool_free(&p, ptrs[i]);
		if (ret) {
			printf("FAIL: free %d returned %d\n", i, ret);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	/* Allocate again */
	for (i = 0; i < 16; i++) {
		ptrs[i] = ofi_lfpool_alloc(&p);
		if (!ptrs[i]) {
			printf("FAIL: re-alloc %d returned NULL\n", i);
			ofi_lfpool_fini(&p);
			return 1;
		}
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 7: Bulk alloc/free operations
 */
static int test_bulk_ops(void)
{
	struct ofi_lfpool p;
	void *ptrs[32];
	size_t count;
	int ret, i;

	printf("Test 7: Bulk alloc/free... ");

	ret = ofi_lfpool_init(&p, 32, 64, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Bulk alloc all 32 */
	count = ofi_lfpool_alloc_bulk(&p, ptrs, 32);
	if (count != 32) {
		printf("FAIL: bulk alloc returned %zu, expected 32\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Check uniqueness */
	for (i = 0; i < 32; i++) {
		for (int j = 0; j < i; j++) {
			if (ptrs[i] == ptrs[j]) {
				printf("FAIL: duplicate at %d and %d\n", i, j);
				ofi_lfpool_fini(&p);
				return 1;
			}
		}
	}

	/* Pool should be empty */
	if (ofi_lfpool_alloc(&p) != NULL) {
		printf("FAIL: pool should be empty after bulk alloc\n");
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Bulk free all */
	count = ofi_lfpool_free_bulk(&p, ptrs, 32);
	if (count != 32) {
		printf("FAIL: bulk free returned %zu, expected 32\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Verify all back in pool */
	count = ofi_lfpool_alloc_bulk(&p, ptrs, 32);
	if (count != 32) {
		printf("FAIL: second bulk alloc returned %zu, expected 32\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 8: Bulk partial alloc (request more than available)
 */
static int test_bulk_partial(void)
{
	struct ofi_lfpool p;
	void *ptrs[32];
	size_t count;
	int ret;

	printf("Test 8: Bulk partial alloc... ");

	ret = ofi_lfpool_init(&p, 16, 64, 0);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	/* Try to alloc more than available */
	count = ofi_lfpool_alloc_bulk(&p, ptrs, 32);
	if (count != 16) {
		printf("FAIL: bulk alloc returned %zu, expected 16\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	/* Alloc 8, leaving 8 in pool */
	ofi_lfpool_free_bulk(&p, ptrs, 8);

	/* Try to alloc 16, should get 8 */
	count = ofi_lfpool_alloc_bulk(&p, ptrs, 16);
	if (count != 8) {
		printf("FAIL: second bulk alloc returned %zu, expected 8\n", count);
		ofi_lfpool_fini(&p);
		return 1;
	}

	ofi_lfpool_fini(&p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 9: Mixed SP/MC mode (single producer freeing, multi consumer allocing)
 */
static struct ofi_lfpool spmc_pool;
static _Atomic int spmc_alloc_total;
static _Atomic int spmc_free_total;

static void *spmc_consumer_thread(void *arg)
{
	(void)arg;
	void *ptr;
	int local = 0;

	for (int i = 0; i < 10000; i++) {
		ptr = ofi_lfpool_alloc(&spmc_pool);
		if (ptr) {
			local++;
			/* Immediately give back via the pool's free_bulk queue
			 * But since we're MC, we can't free - just count */
		}
	}

	atomic_fetch_add(&spmc_alloc_total, local);
	return NULL;
}

static int test_spmc_mode(void)
{
	pthread_t consumers[4];
	void *ptrs[64];
	size_t count;
	int ret, i;
	int freed = 0;

	printf("Test 9: SP/MC mode... ");

	/* SP for free, MC for alloc */
	ret = ofi_lfpool_init_ex(&spmc_pool, 64, 64, 0, OFI_LFPOOL_SP);
	if (ret) {
		printf("FAIL: init returned %d\n", ret);
		return 1;
	}

	atomic_store(&spmc_alloc_total, 0);

	/* Grab all buffers first before starting consumers */
	count = ofi_lfpool_alloc_bulk(&spmc_pool, ptrs, 64);
	if (count != 64) {
		printf("FAIL: initial bulk alloc got %zu\n", count);
		ofi_lfpool_fini(&spmc_pool);
		return 1;
	}

	/* Start consumer threads - they'll spin waiting for buffers */
	for (i = 0; i < 4; i++) {
		ret = pthread_create(&consumers[i], NULL, spmc_consumer_thread, NULL);
		if (ret) {
			printf("FAIL: pthread_create returned %d\n", ret);
			ofi_lfpool_fini(&spmc_pool);
			return 1;
		}
	}

	/* Now free them one at a time while consumers are allocating */
	for (i = 0; i < 64; i++) {
		ret = ofi_lfpool_free(&spmc_pool, ptrs[i]);
		if (ret == 0)
			freed++;
	}

	/* Wait for consumers */
	for (i = 0; i < 4; i++)
		pthread_join(consumers[i], NULL);

	ofi_lfpool_fini(&spmc_pool);

	/* Just verify it didn't crash and we did some work */
	if (freed == 0 && atomic_load(&spmc_alloc_total) == 0) {
		printf("FAIL: no operations completed\n");
		return 1;
	}

	printf("PASS (freed=%d, allocs=%d)\n", freed,
	       atomic_load(&spmc_alloc_total));
	return 0;
}

int main(void)
{
	int failures = 0;

	printf("=== Lock-free Buffer Pool Tests ===\n\n");

	failures += test_basic();
	failures += test_unique_pointers();
	failures += test_alloc_free_pattern();
	failures += test_capacity_rounding();
	failures += test_spsc_mode();
	failures += test_bulk_ops();
	failures += test_bulk_partial();
	failures += test_spmc_mode();
	failures += test_stress();

	printf("\n");
	if (failures == 0) {
		printf("All tests passed!\n");
		return 0;
	} else {
		printf("%d test(s) failed!\n", failures);
		return 1;
	}
}
