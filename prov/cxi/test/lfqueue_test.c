/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

/*
 * lfqueue_test.c - Tests for CXI lock-free data structures
 *
 * Build: gcc -O2 -pthread -I../include -I../../../../include lfqueue_test.c -o lfqueue_test
 * Run: ./lfqueue_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <stdatomic.h>

#include "cxip/lfqueue.h"

#define NUM_THREADS 128
#define OPS_PER_THREAD 100000
#define POOL_CAPACITY 8192

static struct cxip_req_id_pool *pool;
static _Atomic int total_allocs;
static _Atomic int total_frees;
static _Atomic int total_fails;

/*
 * Test 1: Basic single-threaded alloc/free
 */
static int test_basic(void)
{
	struct cxip_req_id_pool *p;
	int ids[16];
	int i, ret;
	void *req;

	printf("Test 1: Basic alloc/free... ");

	ret = cxip_req_id_pool_create(16, &p);
	if (ret) {
		printf("FAIL: create returned %d\n", ret);
		return 1;
	}

	/* Allocate some IDs */
	for (i = 0; i < 8; i++) {
		/* Use (void*)(i+1) as fake request pointer */
		ids[i] = cxip_req_id_alloc(p, (void *)(uintptr_t)(i + 100));
		if (ids[i] <= 0) {
			printf("FAIL: alloc %d returned %d\n", i, ids[i]);
			cxip_req_id_pool_destroy(p);
			return 1;
		}
	}

	/* Verify lookups work */
	for (i = 0; i < 8; i++) {
		req = cxip_req_id_lookup(p, ids[i]);
		if (req != (void *)(uintptr_t)(i + 100)) {
			printf("FAIL: lookup %d returned %p, expected %p\n",
			       i, req, (void *)(uintptr_t)(i + 100));
			cxip_req_id_pool_destroy(p);
			return 1;
		}
	}

	/* Free all */
	for (i = 0; i < 8; i++) {
		req = cxip_req_id_free(p, ids[i]);
		if (req != (void *)(uintptr_t)(i + 100)) {
			printf("FAIL: free %d returned %p, expected %p\n",
			       i, req, (void *)(uintptr_t)(i + 100));
			cxip_req_id_pool_destroy(p);
			return 1;
		}
	}

	/* Verify lookups return NULL after free */
	for (i = 0; i < 8; i++) {
		req = cxip_req_id_lookup(p, ids[i]);
		if (req != NULL) {
			printf("FAIL: lookup after free %d returned %p\n",
			       i, req);
			cxip_req_id_pool_destroy(p);
			return 1;
		}
	}

	cxip_req_id_pool_destroy(p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 2: Unique IDs
 */
static int test_unique_ids(void)
{
	struct cxip_req_id_pool *p;
	int ids[32];
	int i, j, ret;

	printf("Test 2: Unique IDs... ");

	ret = cxip_req_id_pool_create(64, &p);
	if (ret) {
		printf("FAIL: create returned %d\n", ret);
		return 1;
	}

	/* Allocate and check uniqueness */
	for (i = 0; i < 32; i++) {
		ids[i] = cxip_req_id_alloc(p, (void *)(uintptr_t)(i + 1));
		if (ids[i] <= 0) {
			printf("FAIL: alloc %d returned %d\n", i, ids[i]);
			cxip_req_id_pool_destroy(p);
			return 1;
		}

		for (j = 0; j < i; j++) {
			if (ids[j] == ids[i]) {
				printf("FAIL: duplicate ID at %d and %d\n", j, i);
				cxip_req_id_pool_destroy(p);
				return 1;
			}
		}
	}

	cxip_req_id_pool_destroy(p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 3: Pool exhaustion
 */
static int test_exhaustion(void)
{
	struct cxip_req_id_pool *p;
	int count = 0;
	int id;
	int ret;

	printf("Test 3: Pool exhaustion... ");

	/* Small pool - capacity 16 means 15 usable (ID 0 is invalid) */
	ret = cxip_req_id_pool_create(16, &p);
	if (ret) {
		printf("FAIL: create returned %d\n", ret);
		return 1;
	}

	/* Allocate until exhaustion */
	while ((id = cxip_req_id_alloc(p, (void *)(uintptr_t)(count + 1))) > 0) {
		count++;
		if (count > 100) {
			printf("FAIL: allocated too many (%d)\n", count);
			cxip_req_id_pool_destroy(p);
			return 1;
		}
	}

	/* Should have allocated 15 (16 - 1 for invalid ID 0) */
	if (count != 15) {
		printf("FAIL: allocated %d, expected 15\n", count);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	/* Next alloc should fail */
	if (id != -FI_EAGAIN) {
		printf("FAIL: exhausted alloc returned %d, expected -FI_EAGAIN\n", id);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	cxip_req_id_pool_destroy(p);
	printf("PASS\n");
	return 0;
}

/*
 * Test 4: Multi-threaded stress test
 */
static void *stress_thread(void *arg)
{
	int thread_id = *(int *)arg;
	int local_alloc = 0, local_free = 0, local_fail = 0;
	int i, id;
	void *req;

	(void)thread_id;

	for (i = 0; i < OPS_PER_THREAD; i++) {
		/* Use a unique "request" pointer per operation */
		void *fake_req = (void *)(uintptr_t)((thread_id << 20) | i | 0x1000);

		id = cxip_req_id_alloc(pool, fake_req);
		if (id > 0) {
			local_alloc++;

			/* Verify lookup returns our pointer */
			req = cxip_req_id_lookup(pool, id);
			if (req != fake_req) {
				fprintf(stderr, "ERROR: lookup mismatch!\n");
			}

			/* Free it */
			req = cxip_req_id_free(pool, id);
			if (req == fake_req) {
				local_free++;
			}
		} else {
			local_fail++;
		}
	}

	atomic_fetch_add(&total_allocs, local_alloc);
	atomic_fetch_add(&total_frees, local_free);
	atomic_fetch_add(&total_fails, local_fail);

	return NULL;
}

static int test_stress(void)
{
	pthread_t threads[NUM_THREADS];
	int thread_ids[NUM_THREADS];
	int i, ret;

	printf("Test 4: Multi-threaded stress (%d threads, %d ops each)... ",
	       NUM_THREADS, OPS_PER_THREAD);
	fflush(stdout);

	ret = cxip_req_id_pool_create(POOL_CAPACITY, &pool);
	if (ret) {
		printf("FAIL: create returned %d\n", ret);
		return 1;
	}

	atomic_store(&total_allocs, 0);
	atomic_store(&total_frees, 0);
	atomic_store(&total_fails, 0);

	/* Start threads */
	for (i = 0; i < NUM_THREADS; i++) {
		thread_ids[i] = i;
		ret = pthread_create(&threads[i], NULL, stress_thread, &thread_ids[i]);
		if (ret) {
			printf("FAIL: pthread_create returned %d\n", ret);
			cxip_req_id_pool_destroy(pool);
			return 1;
		}
	}

	/* Wait for completion */
	for (i = 0; i < NUM_THREADS; i++)
		pthread_join(threads[i], NULL);

	int allocs = atomic_load(&total_allocs);
	int frees = atomic_load(&total_frees);
	int fails = atomic_load(&total_fails);

	/* Verify allocs == frees */
	if (allocs != frees) {
		printf("FAIL: alloc/free mismatch (%d allocs, %d frees)\n",
		       allocs, frees);
		cxip_req_id_pool_destroy(pool);
		return 1;
	}

	/* Verify pool is empty */
	uint32_t remaining = cxip_req_id_count(pool);
	if (remaining != 0) {
		printf("FAIL: pool has %u remaining allocations\n", remaining);
		cxip_req_id_pool_destroy(pool);
		return 1;
	}

	cxip_req_id_pool_destroy(pool);

	printf("PASS (allocs=%d, fails=%d)\n", allocs, fails);
	return 0;
}

/*
 * Test 5: Invalid ID handling
 */
static int test_invalid_ids(void)
{
	struct cxip_req_id_pool *p;
	void *req;
	int ret;

	printf("Test 5: Invalid ID handling... ");

	ret = cxip_req_id_pool_create(16, &p);
	if (ret) {
		printf("FAIL: create returned %d\n", ret);
		return 1;
	}

	/* Lookup with invalid IDs should return NULL */
	req = cxip_req_id_lookup(p, 0);  /* ID 0 is invalid */
	if (req != NULL) {
		printf("FAIL: lookup(0) returned %p\n", req);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	req = cxip_req_id_lookup(p, -1);  /* Negative ID */
	if (req != NULL) {
		printf("FAIL: lookup(-1) returned %p\n", req);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	req = cxip_req_id_lookup(p, 100);  /* Beyond capacity */
	if (req != NULL) {
		printf("FAIL: lookup(100) returned %p\n", req);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	/* Free with invalid IDs should return NULL */
	req = cxip_req_id_free(p, 0);
	if (req != NULL) {
		printf("FAIL: free(0) returned %p\n", req);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	req = cxip_req_id_free(p, -1);
	if (req != NULL) {
		printf("FAIL: free(-1) returned %p\n", req);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	/* Alloc with NULL request should fail */
	ret = cxip_req_id_alloc(p, NULL);
	if (ret != -FI_EINVAL) {
		printf("FAIL: alloc(NULL) returned %d, expected -FI_EINVAL\n", ret);
		cxip_req_id_pool_destroy(p);
		return 1;
	}

	cxip_req_id_pool_destroy(p);
	printf("PASS\n");
	return 0;
}

int main(void)
{
	int failures = 0;

	printf("=== CXI Lock-Free Queue Tests ===\n\n");

	failures += test_basic();
	failures += test_unique_ids();
	failures += test_exhaustion();
	failures += test_invalid_ids();
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
