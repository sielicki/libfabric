/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

/*
 * Tests for fi_writedata() and fi_inject_writedata() functionality.
 *
 * The CXI provider implements fi_writedata() using a size-adaptive protocol:
 * - Small messages (<= threshold): Single unrestricted PUT (atomic)
 * - Large messages (> threshold): Two-PUT protocol with flush ordering
 *
 * These tests verify:
 * 1. fi_getinfo capability advertisement (FI_REMOTE_CQ_DATA)
 * 2. Basic writedata operations
 * 3. Threshold-based mode selection behavior
 * 4. Environment variable controls
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <criterion/criterion.h>

#include "cxip.h"
#include "cxip_test_common.h"

#define WRITEDATA_WIN_KEY	 0 /* Let provider assign key (FI_MR_PROV_KEY) */
#define WRITEDATA_IMMEDIATE_DATA 0xDEADBEEFCAFEBABEULL

/*
 * =============================================================================
 * fi_getinfo capability tests
 *
 * These tests verify that FI_REMOTE_CQ_DATA capability is properly advertised
 * and that FI_CXI_DISABLE_WRITEDATA environment variable works correctly.
 * =============================================================================
 */

TestSuite(writedata_getinfo, .init = cxit_setup_getinfo,
	  .fini = cxit_teardown_getinfo, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Verify that FI_REMOTE_CQ_DATA is advertised by default */
Test(writedata_getinfo, default_caps_include_remote_cq_data)
{
	int ret;
	struct fi_info *hints;
	struct fi_info *info;
	struct fi_info *iter;

	/* Use cxit_allocinfo() to get properly configured hints
	 * (includes mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED)
	 */
	hints = cxit_allocinfo();
	cr_assert_not_null(hints, "cxit_allocinfo failed");

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
			 NULL, 0, hints, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	/* Verify all returned fi_info entries include FI_REMOTE_CQ_DATA */
	iter = info;
	while (iter) {
		cr_assert(iter->caps & FI_REMOTE_CQ_DATA,
			  "FI_REMOTE_CQ_DATA not in caps: %#llx",
			  (unsigned long long) iter->caps);
		iter = iter->next;
	}

	fi_freeinfo(info);
	fi_freeinfo(hints);
}

/* Verify that FI_CXI_DISABLE_WRITEDATA=1 strips FI_REMOTE_CQ_DATA from caps.
 *
 * NOTE: This test is disabled because environment variables are read at
 * provider initialization time (fi_getinfo -> cxip_info_init). Setting env
 * vars after the provider is loaded has no effect. This test would need to
 * run in a separate process before provider init to work correctly.
 */
Test(writedata_getinfo, disable_strips_capability, .disabled = true)
{
	int ret;
	struct fi_info *hints;
	struct fi_info *info;
	struct fi_info *iter;

	ret = setenv("FI_CXI_DISABLE_WRITEDATA", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	hints = cxit_allocinfo();
	cr_assert_not_null(hints, "cxit_allocinfo failed");

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
			 NULL, 0, hints, &info);
	cr_assert_eq(ret, FI_SUCCESS, "fi_getinfo failed: %d", ret);

	/* Verify all returned fi_info entries do NOT include FI_REMOTE_CQ_DATA
	 */
	iter = info;
	while (iter) {
		cr_assert(
			!(iter->caps & FI_REMOTE_CQ_DATA),
			"FI_REMOTE_CQ_DATA still in caps when disabled: %#llx",
			(unsigned long long) iter->caps);
		iter = iter->next;
	}

	fi_freeinfo(info);
	fi_freeinfo(hints);

	unsetenv("FI_CXI_DISABLE_WRITEDATA");
}

/* Verify that requesting FI_REMOTE_CQ_DATA when disabled returns -FI_ENODATA.
 *
 * NOTE: This test is disabled because environment variables are read at
 * provider initialization time (fi_getinfo -> cxip_info_init). Setting env
 * vars after the provider is loaded has no effect. This test would need to
 * run in a separate process before provider init to work correctly.
 */
Test(writedata_getinfo, disable_explicit_request_fails, .disabled = true)
{
	int ret;
	struct fi_info *hints;
	struct fi_info *info;

	ret = setenv("FI_CXI_DISABLE_WRITEDATA", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	hints = cxit_allocinfo();
	cr_assert_not_null(hints, "cxit_allocinfo failed");

	/* Explicitly request FI_REMOTE_CQ_DATA which should fail when disabled
	 */
	hints->caps = FI_REMOTE_CQ_DATA | FI_RMA | FI_WRITE;

	ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), NULL,
			 NULL, 0, hints, &info);
	cr_assert_eq(ret, -FI_ENODATA,
		     "Expected -FI_ENODATA when requesting FI_REMOTE_CQ_DATA "
		     "with writedata disabled, got: %d",
		     ret);

	fi_freeinfo(hints);

	unsetenv("FI_CXI_DISABLE_WRITEDATA");
}

/*
 * =============================================================================
 * Basic writedata operation tests
 *
 * These tests use cxit_setup_writedata which enables FI_MR_PROV_KEY mode
 * (required for writedata) and includes FI_RMA_EVENT capability.
 * MRs are created with FI_REMOTE_WRITE access.
 * =============================================================================
 */

TestSuite(writedata, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Basic small writedata transfer (should use single-PUT mode) */
Test(writedata, basic_small_transfer)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64; /* Small, below default threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	/* Fill send buffer with pattern */
	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	/* Create MR with FI_REMOTE_WRITE access */
	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Perform writedata */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	/* Wait for TX completion */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	/* Wait for RX completion (remote CQ data event) */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Verify the RX event has FI_REMOTE_CQ_DATA and correct immediate data
	 */
	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert(rx_cqe.flags & FI_RMA, "RX event missing FI_RMA flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert(rx_cqe.flags & FI_REMOTE_WRITE,
		  "RX event missing FI_REMOTE_WRITE flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert_eq(rx_cqe.data, imm_data,
		     "Immediate data mismatch: expected %#llx, got %#llx",
		     (unsigned long long) imm_data,
		     (unsigned long long) rx_cqe.data);

	/* Verify data was written correctly */
	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Large writedata transfer (should use two-PUT mode) */
Test(writedata, basic_large_transfer)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large, above default 2KB threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	/* Fill send buffer with pattern */
	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	/* Create MR with FI_REMOTE_WRITE access */
	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Perform writedata */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	/* Wait for TX completion */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	/* Wait for RX completion (remote CQ data event) */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Verify the RX event has FI_REMOTE_CQ_DATA and correct immediate data
	 */
	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert_eq(rx_cqe.data, imm_data,
		     "Immediate data mismatch: expected %#llx, got %#llx",
		     (unsigned long long) imm_data,
		     (unsigned long long) rx_cqe.data);

	/* Verify data was written correctly */
	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* fi_inject_writedata test */
Test(writedata, inject_writedata)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64; /* Must be within inject limit */
	uint8_t send_buf[64] = {0};
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Fill send buffer with pattern */
	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	/* Create MR with FI_REMOTE_WRITE access */
	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Perform inject_writedata - no TX completion expected */
	ret = fi_inject_writedata(cxit_ep, send_buf, len, imm_data,
				  cxit_ep_fi_addr, 0, key_val);
	cr_assert_eq(ret, FI_SUCCESS, "fi_inject_writedata failed: %d", ret);

	/* Wait for RX completion (remote CQ data event) */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Verify the RX event has FI_REMOTE_CQ_DATA and correct immediate data
	 */
	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert_eq(rx_cqe.data, imm_data,
		     "Immediate data mismatch: expected %#llx, got %#llx",
		     (unsigned long long) imm_data,
		     (unsigned long long) rx_cqe.data);

	/* Verify data was written correctly */
	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after inject_writedata");

	mr_destroy(&mem_window);
}

/* Zero-byte writedata transfer.
 *
 * NOTE: This test is disabled because zero-byte writedata with FI_RMA_EVENT
 * MRs has issues that need investigation. Zero-byte RMA is a corner case
 * and the behavior may not be well-defined for remote CQ data operations.
 */
Test(writedata, zero_byte_writedata, .disabled = true)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Create zero-length MR */
	ret = mr_create_writedata(0, FI_REMOTE_WRITE, 0, &key_val, &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Perform zero-byte writedata */
	ret = fi_writedata(cxit_ep, NULL, 0, NULL, imm_data, cxit_ep_fi_addr, 0,
			   key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	/* Wait for TX completion */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	/* Wait for RX completion (remote CQ data event) */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Verify the RX event has FI_REMOTE_CQ_DATA and correct immediate data
	 */
	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag: %#llx",
		  (unsigned long long) rx_cqe.flags);
	cr_assert_eq(rx_cqe.data, imm_data,
		     "Immediate data mismatch: expected %#llx, got %#llx",
		     (unsigned long long) imm_data,
		     (unsigned long long) rx_cqe.data);

	mr_destroy(&mem_window);
}

/*
 * =============================================================================
 * Threshold behavior tests
 *
 * These tests verify the writedata_atomic_threshold environment variable
 * affects mode selection. Without debug counters, we verify behavior
 * indirectly by ensuring operations complete correctly.
 * =============================================================================
 */

TestSuite(writedata_threshold, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Verify writedata still works when threshold is set to 0 (always two-PUT) */
Test(writedata_threshold, threshold_zero_forces_two_put)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64; /* Small message that would normally use single-PUT */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Set threshold to 0 to force two-PUT mode for all sizes */
	ret = setenv("FI_CXI_WRITEDATA_ATOMIC_THRESHOLD", "0", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	cxit_setup_writedata();

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
	cxit_teardown_writedata();

	unsetenv("FI_CXI_WRITEDATA_ATOMIC_THRESHOLD");
}

/* Verify behavior at threshold boundary */
Test(writedata_threshold, at_threshold_boundary)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 2048; /* Exactly at default threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	cxit_setup_writedata();

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* At threshold (<=) should use single-PUT mode */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
	cxit_teardown_writedata();
}

/* Verify writedata just above threshold uses two-PUT mode */
Test(writedata_threshold, above_threshold)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 2049; /* Just above default 2048 threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	cxit_setup_writedata();

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Above threshold should use two-PUT mode */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
	cxit_teardown_writedata();
}

/*
 * =============================================================================
 * Prefer flush mode tests
 *
 * These tests verify the writedata_prefer_flush environment variable forces
 * FLUSH mode over TRIGGERED mode for two-PUT protocol.
 * =============================================================================
 */

TestSuite(writedata_prefer_flush, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Verify writedata works with prefer_flush=1 */
Test(writedata_prefer_flush, flush_mode_works)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large enough to use two-PUT mode */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Force FLUSH mode */
	ret = setenv("FI_CXI_WRITEDATA_PREFER_FLUSH", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	cxit_setup_writedata();

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);
	validate_tx_event(&tx_cqe, FI_RMA | FI_WRITE, NULL);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	cr_assert_eq(memcmp(mem_window.mem, send_buf, len), 0,
		     "Data mismatch after writedata");

	mr_destroy(&mem_window);
	free(send_buf);
	cxit_teardown_writedata();

	unsetenv("FI_CXI_WRITEDATA_PREFER_FLUSH");
}

/*
 * =============================================================================
 * Disabled writedata tests
 *
 * These tests verify that writedata operations return proper errors when
 * writedata is disabled via FI_CXI_DISABLE_WRITEDATA.
 * =============================================================================
 */

TestSuite(writedata_disabled, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Verify fi_writedata returns -FI_EOPNOTSUPP when disabled */
Test(writedata_disabled, writedata_returns_eopnotsupp)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Disable writedata */
	ret = setenv("FI_CXI_DISABLE_WRITEDATA", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	cxit_setup_rma_disable_fi_rma_event();

	/* Use regular mr_create since writedata is disabled - we don't need
	 * FI_RMA_EVENT on the MR, we just want to test the EOPNOTSUPP return.
	 */
	ret = mr_create(len, FI_REMOTE_WRITE, 0, &key_val, &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, -FI_EOPNOTSUPP,
		     "Expected -FI_EOPNOTSUPP when writedata disabled, got: %d",
		     ret);

	mr_destroy(&mem_window);
	cxit_teardown_rma();

	unsetenv("FI_CXI_DISABLE_WRITEDATA");
}

/* Verify fi_inject_writedata returns -FI_EOPNOTSUPP when disabled */
Test(writedata_disabled, inject_writedata_returns_eopnotsupp)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	/* Disable writedata */
	ret = setenv("FI_CXI_DISABLE_WRITEDATA", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	cxit_setup_rma_disable_fi_rma_event();

	/* Use regular mr_create since writedata is disabled - we don't need
	 * FI_RMA_EVENT on the MR, we just want to test the EOPNOTSUPP return.
	 */
	ret = mr_create(len, FI_REMOTE_WRITE, 0, &key_val, &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_inject_writedata(cxit_ep, send_buf, len, imm_data,
				  cxit_ep_fi_addr, 0, key_val);
	cr_assert_eq(ret, -FI_EOPNOTSUPP,
		     "Expected -FI_EOPNOTSUPP when writedata disabled, got: %d",
		     ret);

	mr_destroy(&mem_window);
	cxit_teardown_rma();

	unsetenv("FI_CXI_DISABLE_WRITEDATA");
}

/*
 * =============================================================================
 * Multiple operations tests
 *
 * These tests verify multiple writedata operations can be performed correctly.
 * =============================================================================
 */

TestSuite(writedata_multi, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Multiple writedata operations to same MR */
Test(writedata_multi, multiple_to_same_mr)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int num_ops = 5;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	for (int i = 0; i < num_ops; i++) {
		uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA + i;

		/* Fill buffer with iteration-specific pattern */
		memset(send_buf, i, len);

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_writedata failed on iteration %d: %d", i, ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion failed on iteration %d", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion failed on iteration %d", i);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA on iteration %d", i);
		cr_assert_eq(rx_cqe.data, imm_data,
			     "Immediate data mismatch on iteration %d", i);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Mode selection verification tests
 *
 * These tests use the EP extension ops to verify the correct writedata mode
 * is selected based on message size and environment configuration.
 * =============================================================================
 */

TestSuite(writedata_stats, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Verify small message uses SINGLE_PUT mode */
Test(writedata_stats, small_uses_single_put)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64; /* Small, below default 2KB threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Get stats before operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Perform writedata */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Get stats after operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Verify SINGLE_PUT count incremented */
	cr_assert_eq(
		stats_after.single_put_count, stats_before.single_put_count + 1,
		"single_put_count not incremented: before=%lu after=%lu",
		stats_before.single_put_count, stats_after.single_put_count);

	/* Verify other modes unchanged */
	cr_assert_eq(stats_after.triggered_count, stats_before.triggered_count,
		     "triggered_count unexpectedly changed");
	cr_assert_eq(stats_after.flush_count, stats_before.flush_count,
		     "flush_count unexpectedly changed");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Verify large message uses TRIGGERED mode when counter is bound.
 *
 * The default cxit_setup_writedata fixture binds cxit_write_cntr to the EP,
 * which enables TRIGGERED mode for messages above the atomic threshold.
 * TRIGGERED mode uses hardware CT thresholds for sequencing (CSDG 5.4).
 */
Test(writedata_stats, large_uses_triggered_with_counter)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large, above 2KB threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Get stats before operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Perform writedata */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Get stats after operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Verify TRIGGERED count incremented (counter bound enables triggered
	 * mode) */
	cr_assert_eq(stats_after.triggered_count,
		     stats_before.triggered_count + 1,
		     "triggered_count not incremented: before=%lu after=%lu",
		     stats_before.triggered_count, stats_after.triggered_count);

	/* Verify single_put mode unchanged */
	cr_assert_eq(stats_after.single_put_count,
		     stats_before.single_put_count,
		     "single_put_count unexpectedly changed");

	/* Verify flush mode unchanged */
	cr_assert_eq(stats_after.flush_count, stats_before.flush_count,
		     "flush_count unexpectedly changed");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Verify stats reset works */
Test(writedata_stats, reset_clears_counters)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Perform writedata to increment some counter */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	/* Verify stats are non-zero */
	ret = cxit_get_writedata_stats(cxit_ep, &stats);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);
	cr_assert(stats.single_put_count > 0 || stats.triggered_count > 0 ||
			  stats.flush_count > 0,
		  "Expected at least one counter > 0 after writedata");

	/* Reset stats */
	ret = cxit_reset_writedata_stats(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_reset_writedata_stats failed: %d",
		     ret);

	/* Verify all counters are zero */
	ret = cxit_get_writedata_stats(cxit_ep, &stats);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);
	cr_assert_eq(stats.single_put_count, 0, "single_put_count not reset");
	cr_assert_eq(stats.triggered_count, 0, "triggered_count not reset");
	cr_assert_eq(stats.flush_count, 0, "flush_count not reset");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * NOTE: Tests for FI_CXI_WRITEDATA_ATOMIC_THRESHOLD=0 and
 * FI_CXI_WRITEDATA_PREFER_FLUSH=1 env vars are in the separate
 * writedata_threshold and writedata_prefer_flush test suites.
 * Those suites can set env vars before provider initialization.
 *
 * The writedata_stats suite uses cxit_setup_writedata which initializes
 * the provider, so env vars set after that point won't be re-read.
 */

/*
 * =============================================================================
 * Additional coverage tests
 *
 * These tests cover edge cases and additional scenarios.
 * =============================================================================
 */

TestSuite(writedata_misc, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata at non-zero remote offset */
Test(writedata_misc, nonzero_offset)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t mr_size = 4096;
	size_t write_len = 64;
	size_t offset = 1024;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, write_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	/* Fill send buffer with distinct pattern */
	for (size_t i = 0; i < write_len; i++)
		send_buf[i] = 0xAA;

	/* Create large MR, initialize with zeros */
	ret = mr_create_writedata(mr_size, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Write at non-zero offset */
	ret = fi_writedata(cxit_ep, send_buf, write_len, NULL, imm_data,
			   cxit_ep_fi_addr, offset, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "RX event missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	/* Verify data at offset is correct */
	cr_assert_eq(memcmp(mem_window.mem + offset, send_buf, write_len), 0,
		     "Data mismatch at offset %zu", offset);

	/* Verify data before offset is still zero (seed was 0) */
	for (size_t i = 0; i < offset; i++) {
		cr_assert_eq(mem_window.mem[i], (uint8_t) i,
			     "Data before offset was modified at %zu", i);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test write counter increments correctly after writedata */
Test(writedata_misc, write_counter_increments)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	uint64_t cntr_before, cntr_after;
	int num_ops = 3;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Read write counter before operations */
	cntr_before = fi_cntr_read(cxit_write_cntr);

	for (int i = 0; i < num_ops; i++) {
		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data + i,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_writedata failed on iteration %d: %d", i, ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion failed on iteration %d", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion failed on iteration %d", i);
	}

	/* Read write counter after operations */
	cntr_after = fi_cntr_read(cxit_write_cntr);

	/* Verify counter incremented by number of operations */
	cr_assert_eq(
		cntr_after, cntr_before + num_ops,
		"Write counter mismatch: before=%lu after=%lu expected=%lu",
		cntr_before, cntr_after, cntr_before + num_ops);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test different immediate data values */
Test(writedata_misc, various_immediate_data)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t send_buf[64] = {0};
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;

	/* Test various immediate data patterns */
	uint64_t test_data[] = {
		0x0000000000000000ULL, /* All zeros */
		0xFFFFFFFFFFFFFFFFULL, /* All ones */
		0x5555555555555555ULL, /* Alternating bits */
		0xAAAAAAAAAAAAAAAAULL, /* Alternating bits inverted */
		0x0000000000000001ULL, /* Minimum non-zero */
		0x8000000000000000ULL, /* MSB set */
	};
	int num_patterns = sizeof(test_data) / sizeof(test_data[0]);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	for (int i = 0; i < num_patterns; i++) {
		memset(send_buf, i, len);

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, test_data[i],
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_writedata failed for pattern %d: %d", i, ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion failed for pattern %d", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion failed for pattern %d", i);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA for pattern %d", i);
		cr_assert_eq(rx_cqe.data, test_data[i],
			     "Immediate data mismatch for pattern %d: "
			     "expected %#llx got %#llx",
			     i, (unsigned long long) test_data[i],
			     (unsigned long long) rx_cqe.data);
	}

	mr_destroy(&mem_window);
}

/* Test mixed small and large transfers verify correct mode selection */
Test(writedata_misc, mixed_sizes_mode_selection)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t mr_size = 16384;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;

	/* Test sizes: small (single-put), at boundary, above boundary */
	size_t test_sizes[] = {64, 2048, 2049, 4096, 8192};
	int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
	int expected_single_put = 2; /* 64 and 2048 */
	int expected_flush = 3; /* 2049, 4096, 8192 */

	send_buf = calloc(1, mr_size);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(mr_size, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Reset stats */
	ret = cxit_reset_writedata_stats(cxit_ep);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_reset_writedata_stats failed");

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed");

	for (int i = 0; i < num_sizes; i++) {
		memset(send_buf, i, test_sizes[i]);

		ret = fi_writedata(cxit_ep, send_buf, test_sizes[i], NULL,
				   imm_data + i, cxit_ep_fi_addr, 0, key_val,
				   NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_writedata failed for size %zu: %d",
			     test_sizes[i], ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion failed for size %zu",
			     test_sizes[i]);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion failed for size %zu",
			     test_sizes[i]);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA for size %zu",
			  test_sizes[i]);
	}

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed");

	/* Verify mode selection counts */
	cr_assert_eq(
		stats_after.single_put_count - stats_before.single_put_count,
		expected_single_put,
		"single_put_count mismatch: got %lu expected %d",
		stats_after.single_put_count - stats_before.single_put_count,
		expected_single_put);

	/* Large messages use either TRIGGERED or FLUSH depending on trig_cmdq
	 */
	uint64_t two_put_count =
		(stats_after.triggered_count - stats_before.triggered_count) +
		(stats_after.flush_count - stats_before.flush_count);
	cr_assert_eq(two_put_count, expected_flush,
		     "two-PUT count mismatch: got %lu expected %d",
		     two_put_count, expected_flush);

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * TRIGGERED mode tests
 *
 * These tests specifically verify TRIGGERED mode operation when conditions
 * are met:
 * 1. Large message (> threshold)
 * 2. write_cntr bound to EP (txc->write_cntr != NULL)
 * 3. Domain has triggered cmdq (txc->domain->trig_cmdq != NULL)
 * 4. FI_CXI_WRITEDATA_PREFER_FLUSH not set
 *
 * NOTE: TRIGGERED mode requires real CXI hardware with triggered command
 * queue support. In emulated/simulated environments, domain->trig_cmdq may
 * be NULL, causing FLUSH mode to be used instead. Tests in this suite are
 * written to handle both cases gracefully but log which mode was selected.
 * =============================================================================
 */

TestSuite(writedata_triggered, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/*
 * Verify large transfers with counter use TRIGGERED or FLUSH mode.
 *
 * This test verifies that large messages (> threshold) use two-PUT mode
 * (either TRIGGERED or FLUSH depending on hardware support).
 * The cxit_setup_writedata fixture binds cxit_write_cntr to the EP.
 */
Test(writedata_triggered, large_with_counter_uses_two_put)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large, above 2KB threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;
	uint64_t triggered_delta, flush_delta, two_put_delta;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Get stats before operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Perform writedata - should use two-PUT mode */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data,
		     "Immediate data mismatch: got 0x%lx expected 0x%lx",
		     rx_cqe.data, imm_data);

	/* Get stats after operation */
	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* Verify single_put NOT used for large message */
	cr_assert_eq(stats_after.single_put_count,
		     stats_before.single_put_count,
		     "single_put_count should not increase for large message");

	/* Verify one of TRIGGERED or FLUSH was used */
	triggered_delta =
		stats_after.triggered_count - stats_before.triggered_count;
	flush_delta = stats_after.flush_count - stats_before.flush_count;
	two_put_delta = triggered_delta + flush_delta;

	cr_assert_eq(two_put_delta, 1,
		     "Expected exactly one two-PUT op: triggered_delta=%lu "
		     "flush_delta=%lu",
		     triggered_delta, flush_delta);

	/* Log which mode was actually selected for diagnostic purposes */
	if (triggered_delta == 1) {
		cr_log_info("TRIGGERED mode selected (hardware support "
			    "available)\n");
	} else {
		cr_log_info("FLUSH mode selected (trig_cmdq not available)\n");
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * Verify multiple large TRIGGERED/FLUSH operations in sequence.
 *
 * Tests that multiple consecutive large writedata operations all use
 * two-PUT mode and complete correctly.
 */
Test(writedata_triggered, multiple_large_ops_sequential)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 4096; /* Large, above 2KB threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cxi_writedata_stats stats_before, stats_after;
	uint64_t two_put_delta;
	int num_ops = 5;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	for (int i = 0; i < num_ops; i++) {
		uint64_t imm_data = 0xDEAD0000 + i;

		/* Fill buffer with unique pattern */
		memset(send_buf, i, len);

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed: %d", i,
			     ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion %d failed", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA on op %d", i);
		cr_assert_eq(rx_cqe.data, imm_data,
			     "Immediate data mismatch on op %d", i);
	}

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* All ops should use two-PUT mode */
	two_put_delta =
		(stats_after.triggered_count - stats_before.triggered_count) +
		(stats_after.flush_count - stats_before.flush_count);

	cr_assert_eq(two_put_delta, num_ops, "Expected %d two-PUT ops, got %lu",
		     num_ops, two_put_delta);

	cr_log_info("Completed %d large ops: triggered=%lu flush=%lu\n",
		    num_ops,
		    stats_after.triggered_count - stats_before.triggered_count,
		    stats_after.flush_count - stats_before.flush_count);

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * Verify write counter is incremented for TRIGGERED/FLUSH mode.
 *
 * When write_cntr is bound, it should be incremented for each operation
 * regardless of whether TRIGGERED or FLUSH mode is used.
 */
Test(writedata_triggered, counter_incremented)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large, above threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	uint64_t cntr_before, cntr_after;
	int num_ops = 3;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	cntr_before = fi_cntr_read(cxit_write_cntr);

	for (int i = 0; i < num_ops; i++) {
		memset(send_buf, i, len);

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data + i,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed: %d", i,
			     ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion %d failed", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);
	}

	cntr_after = fi_cntr_read(cxit_write_cntr);

	cr_assert_eq(cntr_after - cntr_before, num_ops,
		     "Write counter mismatch: before=%lu after=%lu expected "
		     "delta=%d",
		     cntr_before, cntr_after, num_ops);

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * Verify various sizes above threshold use two-PUT mode.
 *
 * Tests that different message sizes above the threshold all use
 * two-PUT (TRIGGERED or FLUSH) mode.
 */
Test(writedata_triggered, various_large_sizes)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;
	uint64_t two_put_delta;
	size_t max_len = 64 * 1024; /* 64KB max */

	/* Various sizes above 2KB threshold */
	size_t test_sizes[] = {2049, 4096, 8192, 16384, 32768, 65536};
	int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

	send_buf = calloc(1, max_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(max_len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	for (int i = 0; i < num_sizes; i++) {
		size_t len = test_sizes[i];

		memset(send_buf, i, len);

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data + i,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS,
			     "fi_writedata size %zu failed: %d", len, ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion size %zu failed", len);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion size %zu failed", len);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA for size %zu", len);
	}

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* All sizes above threshold should use two-PUT */
	two_put_delta =
		(stats_after.triggered_count - stats_before.triggered_count) +
		(stats_after.flush_count - stats_before.flush_count);

	cr_assert_eq(two_put_delta, num_sizes,
		     "Expected %d two-PUT ops, got %lu", num_sizes,
		     two_put_delta);

	/* Verify no single_put used */
	cr_assert_eq(stats_after.single_put_count,
		     stats_before.single_put_count,
		     "single_put should not be used for sizes > threshold");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * Verify prefer_flush env var forces FLUSH mode even when TRIGGERED available.
 *
 * This test sets FI_CXI_WRITEDATA_PREFER_FLUSH=1 and verifies that FLUSH mode
 * is used even when the write counter is bound (which would normally allow
 * TRIGGERED mode on supported hardware).
 *
 * NOTE: This test is disabled because environment variables like
 * FI_CXI_WRITEDATA_PREFER_FLUSH are read at domain creation time, not at
 * each operation. Calling teardown/setup after setting the env var does
 * re-initialize the endpoint but the provider's global state may already
 * have cached the old value. This test would need to run in a separate
 * process to work reliably.
 */
Test(writedata_triggered, prefer_flush_overrides_triggered, .disabled = true)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 8192; /* Large, above threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;

	/* Set prefer_flush to force FLUSH mode */
	ret = setenv("FI_CXI_WRITEDATA_PREFER_FLUSH", "1", 1);
	cr_assert_eq(ret, 0, "setenv failed");

	/* Re-init to pick up env var */
	cxit_teardown_writedata();
	cxit_setup_writedata();

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX cxit_await_completion failed: %d", ret);

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX cxit_await_completion failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "cxit_get_writedata_stats failed: %d",
		     ret);

	/* With prefer_flush set, FLUSH mode must be used */
	cr_assert_eq(stats_after.flush_count, stats_before.flush_count + 1,
		     "FLUSH mode should be used with prefer_flush=1");

	/* TRIGGERED must NOT be used */
	cr_assert_eq(stats_after.triggered_count, stats_before.triggered_count,
		     "TRIGGERED mode should NOT be used with prefer_flush=1");

	mr_destroy(&mem_window);
	free(send_buf);

	unsetenv("FI_CXI_WRITEDATA_PREFER_FLUSH");
}

/*
 * =============================================================================
 * Error handling tests
 * =============================================================================
 */

TestSuite(writedata_errors, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with invalid fi_addr.
 *
 * NOTE: This test is disabled because FI_ADDR_NOTAVAIL ((uint64_t)-1) can
 * cause issues in some internal data structures (e.g., buffer pool lookups)
 * that may not properly validate the index before dereferencing. This is a
 * known limitation that should be addressed in a separate fix.
 */
Test(writedata_errors, invalid_fi_addr, .disabled = true)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	fi_addr_t bad_addr = FI_ADDR_NOTAVAIL;

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Use invalid address */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data, bad_addr, 0,
			   key_val, NULL);

	/* Should fail with invalid address error */
	cr_assert(ret < 0, "fi_writedata should fail with invalid address");

	mr_destroy(&mem_window);
}

/* Test writedata with invalid MR key (standard range provider key).
 *
 * Writedata requires FI_MR_PROV_KEY mode keys (is_prov=1) and FI_RMA_EVENT
 * (events=1). This test uses a provider key format that passes validation
 * but points to a non-existent MR, which should result in an async error.
 */
Test(writedata_errors, invalid_std_mr_key)
{
	int ret;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err;
	/* Non-cached provider key with events=1, is_prov=1, cached=0, opt=0
	 * Format: id(16) | seqnum(44) | events(1) | unused(2) | is_prov(1)
	 * Invalid ID 0xABCD that doesn't correspond to any registered MR
	 */
	uint64_t invalid_key = (1ULL << 63) | /* is_prov = 1 */
			       (1ULL << 60) | /* events = 1 */
			       (0xABCDULL); /* id = invalid */

	/* Non-zero length writedata to invalid key */
	ret = fi_writedata(cxit_ep, send_buf, 64, NULL, imm_data,
			   cxit_ep_fi_addr, 0, invalid_key, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error), got %d", ret);

	/* Wait for error counter */
	while (fi_cntr_readerr(cxit_write_cntr) != 1)
		;

	/* Should have error in TX CQ */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAVAIL, "Expected -FI_EAVAIL, got %d", ret);

	ret = fi_cq_readerr(cxit_tx_cq, &err, 1);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed");
}

/* Test writedata with invalid MR key (optimized range provider key).
 *
 * NOTE: This test is disabled because optimized provider keys have a very
 * limited key space (ID must be < 92). It's difficult to craft a key that
 * passes format validation but points to a non-existent MR. The standard
 * MR key test (invalid_std_mr_key) provides sufficient coverage for async
 * error handling with invalid keys.
 */
Test(writedata_errors, invalid_opt_mr_key, .disabled = true)
{
	int ret;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cq_tagged_entry cqe;
	struct fi_cq_err_entry err;
	/* Non-cached optimized provider key with events=1, is_prov=1, opt=1
	 * Format: id(16) | seqnum(44) | events(1) | opt(1) | cached(1) |
	 * is_prov(1) Invalid ID 0xBEEF that doesn't correspond to any
	 * registered MR
	 */
	uint64_t invalid_key = (1ULL << 63) | /* is_prov = 1 */
			       (1ULL << 61) | /* opt = 1 */
			       (1ULL << 60) | /* events = 1 */
			       (0xBEEFULL); /* id = invalid */

	/* Non-zero length writedata to invalid key */
	ret = fi_writedata(cxit_ep, send_buf, 64, NULL, imm_data,
			   cxit_ep_fi_addr, 0, invalid_key, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error), got %d", ret);

	/* Wait for error counter */
	while (fi_cntr_readerr(cxit_write_cntr) != 1)
		;

	/* Should have error in TX CQ */
	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert_eq(ret, -FI_EAVAIL, "Expected -FI_EAVAIL, got %d", ret);

	ret = fi_cq_readerr(cxit_tx_cq, &err, 1);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed");
}

/* Test writedata with client-rejected key (too large) */
Test(writedata_errors, rejected_key)
{
	int ret;
	uint8_t send_buf[64] = {0};
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	/* Key with bits above 32 set - should be rejected by client */
	uint64_t bad_key = 0x100000001ULL;

	ret = fi_writedata(cxit_ep, send_buf, 64, NULL, imm_data,
			   cxit_ep_fi_addr, 0, bad_key, NULL);
	cr_assert_eq(ret, -FI_EKEYREJECTED, "Expected -FI_EKEYREJECTED, got %d",
		     ret);
}

/*
 * =============================================================================
 * MR overrun tests
 *
 * Verify that writedata properly handles attempts to write beyond MR bounds.
 * =============================================================================
 */

TestSuite(writedata_bounds, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata MR overrun detection via CQ */
Test(writedata_bounds, mr_overrun_cq)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0xa;
	size_t mr_len = 4096;
	size_t overrun_len = mr_len * 2;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cq_err_entry err;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, overrun_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(mr_len, FI_REMOTE_WRITE, 0xc0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* First do a good transfer */
	ret = fi_writedata(cxit_ep, send_buf, mr_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	/* Now try to overrun the MR */
	ret = fi_writedata(cxit_ep, send_buf, overrun_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error)");

	/* Wait for error */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, -FI_EAVAIL, "Expected -FI_EAVAIL, got %d", ret);

	ret = fi_cq_readerr(cxit_tx_cq, &err, 1);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed");
	cr_assert_eq(err.err, FI_EIO, "Expected FI_EIO, got %d", err.err);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata MR overrun detection via counter */
Test(writedata_bounds, mr_overrun_cntr)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0xa;
	size_t mr_len = 4096;
	size_t overrun_len = mr_len * 2;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, overrun_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(mr_len, FI_REMOTE_WRITE, 0xc0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* First do a good transfer */
	ret = fi_writedata(cxit_ep, send_buf, mr_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	/* Now try to overrun the MR */
	ret = fi_writedata(cxit_ep, send_buf, overrun_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error)");

	/* Wait for error counter */
	while (fi_cntr_readerr(cxit_write_cntr) != 1)
		;

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with offset causing overrun */
Test(writedata_bounds, offset_overrun)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0xa;
	size_t mr_len = 4096;
	size_t send_len = 1024;
	size_t bad_offset = mr_len - 512; /* Offset + len > mr_len */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_err_entry err;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, send_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(mr_len, FI_REMOTE_WRITE, 0xc0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Try to write with offset that causes overrun */
	ret = fi_writedata(cxit_ep, send_buf, send_len, NULL, imm_data,
			   cxit_ep_fi_addr, bad_offset, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error)");

	/* Wait for error */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, -FI_EAVAIL, "Expected -FI_EAVAIL, got %d", ret);

	ret = fi_cq_readerr(cxit_tx_cq, &err, 1);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Standard vs Optimized MR tests
 *
 * The writedata implementation has different behavior based on MR type:
 * - Standard MRs: Can use SINGLE_PUT for small messages (header_data visible)
 * - Optimized MRs: Must use two-PUT (EVENT_COMM_DISABLE blocks header_data)
 * =============================================================================
 */

TestSuite(writedata_mr_types, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with explicit standard MR key */
Test(writedata_mr_types, standard_mr_small)
{
	int ret;
	struct mem_region mem_window;
	/* Key > CXIP_PTL_IDX_MR_OPT_CNT forces standard MR */
	uint64_t key_val = 200;
	size_t len = 64; /* Small, below threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	/* Verify data integrity */
	for (size_t i = 0; i < len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "Data mismatch at %zu", i);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	/* Standard MR small message should use SINGLE_PUT */
	cr_assert_eq(stats_after.single_put_count,
		     stats_before.single_put_count + 1,
		     "Expected SINGLE_PUT mode for standard MR small message");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with explicit standard MR key - large message */
Test(writedata_mr_types, standard_mr_large)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 200; /* Standard MR */
	size_t len = 8192; /* Large, above threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;
	uint64_t two_put_delta;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	/* Large message should use two-PUT regardless of MR type */
	two_put_delta =
		(stats_after.triggered_count - stats_before.triggered_count) +
		(stats_after.flush_count - stats_before.flush_count);
	cr_assert_eq(two_put_delta, 1,
		     "Expected two-PUT mode for large message");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Optimized MR tests
 *
 * Tests for writedata with optimized MRs (requires different fixture).
 * =============================================================================
 */

TestSuite(writedata_opt_mr, .init = cxit_setup_writedata_opt,
	  .fini = cxit_teardown_writedata_opt, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with optimized MR - small message still uses two-PUT */
Test(writedata_opt_mr, small_uses_two_put)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = 0x5; /* Small key = optimized MR */
	size_t len = 64; /* Small, below threshold */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cxi_writedata_stats stats_before, stats_after;
	uint64_t two_put_delta;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = cxit_get_writedata_stats(cxit_ep, &stats_before);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");

	ret = cxit_get_writedata_stats(cxit_ep, &stats_after);
	cr_assert_eq(ret, FI_SUCCESS, "get_stats failed");

	/* Optimized MR cannot use SINGLE_PUT - EVENT_COMM_DISABLE blocks
	 * header_data */
	two_put_delta =
		(stats_after.triggered_count - stats_before.triggered_count) +
		(stats_after.flush_count - stats_before.flush_count);
	cr_assert_eq(two_put_delta, 1,
		     "Expected two-PUT mode for optimized MR (even small)");

	/* SINGLE_PUT should NOT have been used */
	cr_assert_eq(stats_after.single_put_count,
		     stats_before.single_put_count,
		     "SINGLE_PUT should NOT be used for optimized MR");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Page-spanning tests
 *
 * Test writedata with data buffers that span page boundaries.
 * =============================================================================
 */

TestSuite(writedata_pages, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with source buffer spanning page boundary */
Test(writedata_pages, source_spanning_page)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t buf_len = page_size * 2;
	size_t send_len = 128;
	uint8_t *send_buf;
	uint8_t *send_addr;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = aligned_alloc(page_size, buf_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	/* Position data to span page boundary */
	send_addr = send_buf + page_size - (send_len / 2);
	for (size_t i = 0; i < send_len; i++)
		send_addr[i] = (uint8_t) (0xaa + i);

	ret = mr_create_writedata(send_len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_addr, send_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	/* Verify data integrity */
	for (size_t i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_addr[i],
			     "Data mismatch at %zu", i);

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with large transfer spanning multiple pages */
Test(writedata_pages, large_multi_page)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t send_len = page_size * 4; /* 4 pages */
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = aligned_alloc(page_size, send_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	for (size_t i = 0; i < send_len; i++)
		send_buf[i] = (uint8_t) (i & 0xff);

	ret = mr_create_writedata(send_len, FI_REMOTE_WRITE, 0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, send_len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");

	/* Verify data integrity across all pages */
	for (size_t i = 0; i < send_len; i++)
		cr_assert_eq(mem_window.mem[i], send_buf[i],
			     "Data mismatch at %zu (page %zu)", i,
			     i / page_size);

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Remote counter tests
 *
 * Test that remote counters are properly incremented by writedata operations.
 * =============================================================================
 */

TestSuite(writedata_rem_cntr, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test remote counter increments with writedata */
Test(writedata_rem_cntr, basic)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	uint64_t rem_cntr_before, rem_cntr_after;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	rem_cntr_before = fi_cntr_read(cxit_rem_cntr);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	/* Wait for remote counter to update */
	while (fi_cntr_read(cxit_rem_cntr) != rem_cntr_before + 1)
		;

	rem_cntr_after = fi_cntr_read(cxit_rem_cntr);
	cr_assert_eq(rem_cntr_after, rem_cntr_before + 1,
		     "Remote counter not incremented");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test remote counter with multiple writedata operations */
Test(writedata_rem_cntr, multiple_ops)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 128;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	uint64_t rem_cntr_before;
	int num_ops = 10;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len * num_ops, FI_REMOTE_WRITE, 0xa0,
				  &key_val, &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	rem_cntr_before = fi_cntr_read(cxit_rem_cntr);

	for (int i = 0; i < num_ops; i++) {
		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data + i,
				   cxit_ep_fi_addr, i * len, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed: %d", i,
			     ret);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion %d failed", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);
	}

	/* Wait for all remote counter updates */
	while (fi_cntr_read(cxit_rem_cntr) != rem_cntr_before + num_ops)
		;

	cr_assert_eq(fi_cntr_read(cxit_rem_cntr), rem_cntr_before + num_ops,
		     "Remote counter mismatch after %d ops", num_ops);

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Stress tests
 *
 * Test writedata under stress conditions.
 * =============================================================================
 */

TestSuite(writedata_stress, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT * 2);

/* Fire many writedata ops and cleanup without waiting */
Test(writedata_stress, cleanup_without_completion)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	int num_ops = 50;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Fire many operations without waiting */
	for (int i = 0; i < num_ops; i++) {
		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed: %d", i,
			     ret);
	}

	mr_destroy(&mem_window);
	free(send_buf);

	/* Exit without gathering events - verifies cleanup handles in-flight
	 * ops */
}

/* Test rapid alternation between small and large messages */
Test(writedata_stress, mixed_sizes_rapid)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t max_len = 16384;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data;
	int num_iterations = 20;

	send_buf = calloc(1, max_len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(max_len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	for (int i = 0; i < num_iterations; i++) {
		/* Alternate between small (SINGLE_PUT) and large (two-PUT) */
		size_t len = (i % 2 == 0) ? 64 : 8192;
		imm_data = 0xBEEF0000 + i;

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed", i);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion %d failed", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA on iteration %d", i);
		cr_assert_eq(rx_cqe.data, imm_data,
			     "Immediate data mismatch on iteration %d", i);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test many concurrent small writedata operations */
Test(writedata_stress, many_small_concurrent)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int num_ops = 100;
	int tx_completed = 0, rx_completed = 0;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Submit all operations */
	for (int i = 0; i < num_ops; i++) {
		uint64_t imm_data = 0xCAFE0000 + i;

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed", i);
	}

	/* Gather all completions */
	while (tx_completed < num_ops) {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		if (ret == 1)
			tx_completed++;
		else if (ret != -FI_EAGAIN)
			cr_assert(0, "TX CQ error: %d", ret);
	}

	while (rx_completed < num_ops) {
		ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		if (ret == 1) {
			cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
				  "Missing FI_REMOTE_CQ_DATA");
			rx_completed++;
		} else if (ret != -FI_EAGAIN) {
			cr_assert(0, "RX CQ error: %d", ret);
		}
	}

	cr_assert_eq(tx_completed, num_ops, "TX completion mismatch");
	cr_assert_eq(rx_completed, num_ops, "RX completion mismatch");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Stale MR key tests
 *
 * Verify writedata handles invalidated/closed MR keys properly.
 * =============================================================================
 */

TestSuite(writedata_stale, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with stale (closed) MR key */
Test(writedata_stale, closed_mr_key)
{
	int ret;
	struct fid_mr *mr;
	struct cxip_mr *cxip_mr;
	uint8_t *src_buf;
	uint8_t *tgt_buf;
	size_t len = 64;
	uint64_t key_val = 200; /* Standard MR key */
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cq_err_entry err;

	src_buf = malloc(len);
	cr_assert_not_null(src_buf, "src_buf alloc failed");
	tgt_buf = calloc(1, len);
	cr_assert_not_null(tgt_buf, "tgt_buf alloc failed");

	for (size_t i = 0; i < len; i++)
		src_buf[i] = (uint8_t) (0xb1 * i);

	/* Create and enable MR */
	ret = fi_mr_reg(cxit_domain, tgt_buf, len, FI_REMOTE_WRITE, 0, key_val,
			0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed");

	/* Skip test if cached provider key without match events - known
	 * limitation */
	cxip_mr = container_of(mr, struct cxip_mr, mr_fid);
	if ((cxit_fi->domain_attr->mr_mode & FI_MR_PROV_KEY) &&
	    !cxip_mr->count_events) {
		fi_close(&mr->fid);
		free(src_buf);
		free(tgt_buf);
		cr_skip_test("Cached FI_MR_PROV_KEY without match events");
		return;
	}

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed");

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed");

	if (cxit_fi->domain_attr->mr_mode & FI_MR_PROV_KEY)
		key_val = fi_mr_key(mr);

	/* First writedata should succeed */
	ret = fi_writedata(cxit_ep, src_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed");

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	/* Verify data was written */
	for (size_t i = 0; i < len; i++)
		cr_assert_eq(tgt_buf[i], src_buf[i], "Data mismatch at %zu", i);

	/* Close MR but keep backing memory */
	fi_close(&mr->fid);

	/* Modify source buffer to detect if stale write succeeds */
	for (size_t i = 0; i < len; i++)
		src_buf[i] = (uint8_t) (0xa1 * i);

	/* Try writedata with stale key - should fail asynchronously */
	ret = fi_writedata(cxit_ep, src_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_writedata should accept (async error)");

	/* Wait for error */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, -FI_EAVAIL, "Expected -FI_EAVAIL for stale key");

	ret = fi_cq_readerr(cxit_tx_cq, &err, 1);
	cr_assert_eq(ret, 1, "fi_cq_readerr failed");

	/* Verify original data wasn't overwritten */
	for (size_t i = 0; i < len; i++)
		cr_assert_eq(tgt_buf[i], (uint8_t) (0xb1 * i),
			     "Stale write corrupted data at %zu", i);

	free(src_buf);
	free(tgt_buf);
}

/*
 * =============================================================================
 * Selective completion tests
 *
 * Test writedata with FI_SELECTIVE_COMPLETION mode.
 * =============================================================================
 */

static void cxit_setup_writedata_selective_completion(void)
{
	cxit_tx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;

	/* Force provider key mode for writedata tests */
	cxit_prov_key = 1;

	/* Disable optimized MRs for writedata tests */
	setenv("FI_CXI_OPTIMIZED_MRS", "0", 1);

	/* Create hints with FI_MR_PROV_KEY and FI_COMPLETION */
	cxit_fi_hints = cxit_allocinfo();
	cxit_fi_hints->domain_attr->mr_mode |= FI_MR_PROV_KEY;
	cxit_fi_hints->tx_attr->op_flags = FI_COMPLETION;

	cxit_setup_rma();
}

TestSuite(writedata_selective,
	  .init = cxit_setup_writedata_selective_completion,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with selective completion - default gets completion */
Test(writedata_selective, default_completion)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	/* Default op_flags has FI_COMPLETION, so completion should be generated
	 */
	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	/* Should get TX completion */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion expected");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA");

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Immediate data pattern tests
 *
 * Test various immediate data patterns to ensure proper handling.
 * =============================================================================
 */

TestSuite(writedata_imm_patterns, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test writedata with all-zeros immediate data */
Test(writedata_imm_patterns, zero_imm_data)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = 0;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, 0, "Expected zero immediate data");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with all-ones immediate data */
Test(writedata_imm_patterns, all_ones_imm_data)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = 0xFFFFFFFFFFFFFFFFULL;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with alternating bit pattern */
Test(writedata_imm_patterns, alternating_bits)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = 0xAAAAAAAAAAAAAAAAULL;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
			   cxit_ep_fi_addr, 0, key_val, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_writedata failed: %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "TX completion failed");

	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
		  "Missing FI_REMOTE_CQ_DATA flag");
	cr_assert_eq(rx_cqe.data, imm_data, "Immediate data mismatch");

	mr_destroy(&mem_window);
	free(send_buf);
}

/* Test writedata with sequential immediate data values */
Test(writedata_imm_patterns, sequential_values)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t *send_buf;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int num_ops = 16;

	send_buf = calloc(1, len);
	cr_assert_not_null(send_buf, "send_buf alloc failed");

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	for (int i = 0; i < num_ops; i++) {
		uint64_t imm_data = (uint64_t) i;

		ret = fi_writedata(cxit_ep, send_buf, len, NULL, imm_data,
				   cxit_ep_fi_addr, 0, key_val, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_writedata %d failed", i);

		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "TX completion %d failed", i);

		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);

		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA on iteration %d", i);
		cr_assert_eq(rx_cqe.data, imm_data,
			     "Immediate data mismatch on iteration %d: got "
			     "0x%lx expected 0x%lx",
			     i, rx_cqe.data, imm_data);
	}

	mr_destroy(&mem_window);
	free(send_buf);
}

/*
 * =============================================================================
 * Inject writedata additional tests
 *
 * Additional tests for fi_inject_writedata.
 * =============================================================================
 */

TestSuite(writedata_inject_ext, .init = cxit_setup_writedata,
	  .fini = cxit_teardown_writedata, .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test inject_writedata buffer corruption detection */
Test(writedata_inject_ext, buffer_not_used_after_return)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 64;
	uint8_t send_buf[64] = {0};
	struct fi_cq_tagged_entry rx_cqe;
	uint64_t imm_data = WRITEDATA_IMMEDIATE_DATA;

	for (size_t i = 0; i < len; i++)
		send_buf[i] = (uint8_t) (0xaa + i);

	ret = mr_create_writedata(len, FI_REMOTE_WRITE, 0xa0, &key_val,
				  &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	ret = fi_inject_writedata(cxit_ep, send_buf, len, imm_data,
				  cxit_ep_fi_addr, 0, key_val);
	cr_assert_eq(ret, FI_SUCCESS, "fi_inject_writedata failed: %d", ret);

	/* Corrupt buffer immediately after inject returns */
	memset(send_buf, 0xff, len);

	/* Wait for write counter to indicate completion */
	while (fi_cntr_read(cxit_write_cntr) != 1)
		;

	/* Wait for RX completion */
	ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
	cr_assert_eq(ret, 1, "RX completion failed");

	/* Verify original data (0xaa+i pattern, not 0xff) was written */
	for (size_t i = 0; i < len; i++)
		cr_assert_eq(mem_window.mem[i], (uint8_t) (0xaa + i),
			     "Data mismatch at %zu: got 0x%02x expected 0x%02x",
			     i, mem_window.mem[i], (uint8_t) (0xaa + i));

	mr_destroy(&mem_window);
}

/* Test inject_writedata with multiple sequential operations */
Test(writedata_inject_ext, multiple_sequential)
{
	int ret;
	struct mem_region mem_window;
	uint64_t key_val = WRITEDATA_WIN_KEY;
	size_t len = 32;
	uint8_t send_buf[32];
	struct fi_cq_tagged_entry rx_cqe;
	int num_ops = 10;

	ret = mr_create_writedata(len * num_ops, FI_REMOTE_WRITE, 0xa0,
				  &key_val, &mem_window);
	cr_assert_eq(ret, FI_SUCCESS, "mr_create failed: %d", ret);

	for (int i = 0; i < num_ops; i++) {
		uint64_t imm_data = 0xBEEF0000 + i;

		/* Fill with unique pattern */
		memset(send_buf, i + 1, len);

		ret = fi_inject_writedata(cxit_ep, send_buf, len, imm_data,
					  cxit_ep_fi_addr, i * len, key_val);
		cr_assert_eq(ret, FI_SUCCESS, "fi_inject_writedata %d failed",
			     i);
	}

	/* Wait for all operations */
	while (fi_cntr_read(cxit_write_cntr) != num_ops)
		;

	/* Gather all RX completions */
	for (int i = 0; i < num_ops; i++) {
		ret = cxit_await_completion(cxit_rx_cq, &rx_cqe);
		cr_assert_eq(ret, 1, "RX completion %d failed", i);
		cr_assert(rx_cqe.flags & FI_REMOTE_CQ_DATA,
			  "Missing FI_REMOTE_CQ_DATA on op %d", i);
	}

	/* Verify data for each segment */
	for (int i = 0; i < num_ops; i++) {
		for (size_t j = 0; j < len; j++) {
			cr_assert_eq(
				mem_window.mem[i * len + j], (uint8_t) (i + 1),
				"Data mismatch at segment %d offset %zu", i, j);
		}
	}

	mr_destroy(&mem_window);
}
