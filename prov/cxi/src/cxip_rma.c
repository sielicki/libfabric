/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018,2021-2023 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <ofi.h>
#include <ofi_list.h>

#include "cxip.h"

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * cxip_rma_selective_completion_cb() - RMA selective completion callback.
 */
int cxip_rma_selective_completion_cb(struct cxip_req *req,
				     const union c_event *event)
{
	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		CXIP_WARN("Unexpected %s event: rc=%s\n",
			  cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	int event_rc;

	event_rc = cxi_init_event_rc(event);
	int ret_err;

	ret_err = proverr2errno(event_rc);
	return cxip_cq_req_error(req, 0, ret_err, cxi_event_rc(event), NULL, 0,
				 FI_ADDR_UNSPEC);
}

/*
 * cxip_rma_write_selective_completion_req() - Return request state associated
 * with all RMA write with selective completion transactions on the transmit
 * context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *
cxip_rma_write_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_write_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t) txc->context;
		req->flags = FI_RMA | FI_WRITE;
		req->addr = FI_ADDR_UNSPEC;

		txc->rma_write_selective_completion_req = req;
	}

	return txc->rma_write_selective_completion_req;
}

/*
 * cxip_rma_read_selective_completion_req() - Return request state associated
 * with all RMA read with selective completion transactions on the transmit
 * context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *
cxip_rma_read_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_read_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t) txc->context;
		req->flags = FI_RMA | FI_READ;
		req->addr = FI_ADDR_UNSPEC;

		txc->rma_read_selective_completion_req = req;
	}

	return txc->rma_read_selective_completion_req;
}

/*
 * cxip_rma_cb() - RMA event callback.
 */
static int cxip_rma_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	int ret_err;
	bool success_event = !!(req->flags & FI_COMPLETION);
	struct cxip_txc *txc = req->rma.txc;

	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		TXC_WARN(txc, CXIP_UNEXPECTED_EVENT, cxi_event_to_str(event),
			 cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	if (req->rma.cntr)
		cxip_cntr_progress_dec(req->rma.cntr);

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	if (req->rma.ibuf)
		cxip_txc_ibuf_free(txc, req->rma.ibuf);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				TXC_WARN(txc,
					 "Failed to report completion: %d\n",
					 ret);
		}
	} else {
		ret_err = proverr2errno(event_rc);
		ret = cxip_cq_req_error(req, 0, ret_err, event_rc, NULL, 0,
					FI_ADDR_UNSPEC);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report error: %d\n", ret);
	}

	cxip_txc_otx_reqs_dec(req->rma.txc);
	cxip_evtq_req_free(req);

	return FI_SUCCESS;
}

/*
 * cxip_rma_writedata_cb() - Writedata FLUSH mode completion callback.
 *
 * For FLUSH mode fi_writedata(), we emit two DMA commands:
 * 1. Data PUT (restricted, with flush) - carries the actual data
 * 2. Notification PUT (unrestricted) - carries header_data to target
 *
 * Both operations must complete before we can report completion to the user.
 *
 * CSDG References for ordering:
 * - CSDG 5.1.2.3 (Send Options): "If the FLUSH option is set in a Put or AMO
 *   command, return of a response is delayed until the write is visible on
 *   the target node, i.e. a read on that node will return the new value."
 * - CSDG 5.6.3 (Source Ordering): "Command completion indication may also be
 *   requested by an event posted to an event queue... These events are posted
 *   in the order that responses arrive from the network, not necessarily the
 *   order that commands were issued."
 * - CSDG 2.4.1 (Restricted Commands): "Restricted commands are unordered."
 * - CSDG 5.6.4 (Network Ordering): "The request packets resulting from those
 *   restricted commands may take any path through the network to the
 *   destination."
 *
 * NOTE: Events may arrive out of order (per CSDG 5.6.3), so we cannot assume
 * the first event is from the data PUT. We simply count events and track
 * return codes - the actual order doesn't affect correctness since we wait
 * for both and report error if either failed.
 */
static int cxip_rma_writedata_cb(struct cxip_req *req,
				 const union c_event *event)
{
	int ret;
	int event_rc;
	int ret_err;
	bool success_event = !!(req->flags & FI_COMPLETION);
	struct cxip_txc *txc = req->rma.txc;

	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		TXC_WARN(txc, CXIP_UNEXPECTED_EVENT, cxi_event_to_str(event),
			 cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	event_rc = cxi_init_event_rc(event);

	/* Track event count. Per CSDG 5.6.3: "events are posted in the order
	 * that responses arrive from the network, not necessarily the order
	 * that commands were issued." So we cannot assume which PUT this
	 * event corresponds to - we just need to wait for both and track
	 * if either failed.
	 *
	 * Note: Variable names (writedata_data_rc, writedata_notify_rc) are
	 * for logical clarity but actual mapping is arbitrary due to
	 * out-of-order event delivery.
	 */
	req->rma.writedata_events++;

	if (req->rma.writedata_events == 1) {
		/* First arriving event (could be either PUT) */
		req->rma.writedata_data_rc = event_rc;
		TXC_DBG(txc, "Writedata event 1/2: req=%p rc=%d\n", req,
			event_rc);
	} else {
		/* Second arriving event (could be either PUT) */
		req->rma.writedata_notify_rc = event_rc;
		TXC_DBG(txc, "Writedata event 2/2: req=%p rc=%d\n", req,
			event_rc);
	}

	/* If we haven't received all expected events, wait for more.
	 * Normally expects 2 events (data + notify). If notification PUT
	 * failed to emit, expects only 1 event (data only).
	 */
	if (req->rma.writedata_events < req->rma.writedata_events_expected) {
		return FI_SUCCESS;
	}

	/* All events received - now complete the request */
	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	if (req->rma.cntr)
		cxip_cntr_progress_dec(req->rma.cntr);

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	if (req->rma.ibuf)
		cxip_txc_ibuf_free(txc, req->rma.ibuf);

	/* Report error if either operation failed.
	 * Prioritize data PUT errors over notification PUT errors since
	 * data corruption is more severe than missing remote CQ event.
	 */
	if (req->rma.writedata_data_rc != C_RC_OK) {
		ret_err = proverr2errno(req->rma.writedata_data_rc);
		ret = cxip_cq_req_error(req, 0, ret_err,
					req->rma.writedata_data_rc, NULL, 0,
					FI_ADDR_UNSPEC);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report data error: %d\n", ret);
	} else if (req->rma.writedata_notify_rc != C_RC_OK) {
		ret_err = proverr2errno(req->rma.writedata_notify_rc);
		ret = cxip_cq_req_error(req, 0, ret_err,
					req->rma.writedata_notify_rc, NULL, 0,
					FI_ADDR_UNSPEC);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report notify error: %d\n",
				 ret);
	} else if (success_event) {
		ret = cxip_cq_req_complete(req);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report completion: %d\n", ret);
	}

	/* Decrement outstanding requests for each DMA command that was sent.
	 * Normally 2 (data + notify), but only 1 if notify failed to emit.
	 */
	for (int i = 0; i < req->rma.writedata_events_expected; i++)
		cxip_txc_otx_reqs_dec(req->rma.txc);
	cxip_evtq_req_free(req);

	return FI_SUCCESS;
}

/*
 * cxip_rma_writedata_triggered_cb() - Writedata TRIGGERED mode callback.
 *
 * For triggered mode fi_writedata(), we emit two DMA commands:
 * 1. Data PUT (restricted, flush) - carries the actual data, increments CT
 *    on ACK, has event_success_disable=1
 * 2. Notification PUT (triggered, unrestricted) - triggered when CT reaches
 *    threshold, generates the only completion event
 *
 * CSDG References:
 * - CSDG 5.4 (Triggered Operations): "Issuance of the DMA is deferred until
 *   the success field of the counting event TRIG_CT reaches or exceeds the
 *   specific threshold value of the triggered operation."
 * - CSDG 5.4: "If a counting event has already reached the threshold value
 *   when a triggered operation is processed, that operation is processed
 *   immediately." (This is fine - if CT already reached threshold, the data
 *   PUT ACK already arrived, meaning data is visible due to flush.)
 * - CSDG 5.3 (Counting Events), Table 89: "EVENT_CT_ACK - Counting event is
 *   incremented when Put or non-fetching AMO has completed successfully"
 * - CSDG 5.1.2.3: "EVENT_SUCCESS_DISABLE - Only generate full events if
 *   there is an error"
 *
 * This provides race-free sequencing: the notification PUT is not issued
 * until the data PUT's ACK has incremented the CT, and flush ensures the
 * data is visible at the target before the ACK is sent.
 */
static int cxip_rma_writedata_triggered_cb(struct cxip_req *req,
					   const union c_event *event)
{
	int ret;
	int event_rc;
	int ret_err;
	bool success_event = !!(req->flags & FI_COMPLETION);
	struct cxip_txc *txc = req->rma.txc;

	/* When errors happen, send events can occur before the put/get event.
	 * These events should just be dropped.
	 */
	if (event->hdr.event_type == C_EVENT_SEND) {
		TXC_WARN(txc, CXIP_UNEXPECTED_EVENT, cxi_event_to_str(event),
			 cxi_rc_to_str(cxi_event_rc(event)));
		return FI_SUCCESS;
	}

	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	/* Counter progress tracking for the triggered counter */
	if (req->rma.writedata_trig_cntr)
		cxip_cntr_progress_dec(req->rma.writedata_trig_cntr);

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	if (req->rma.ibuf)
		cxip_txc_ibuf_free(txc, req->rma.ibuf);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				TXC_WARN(txc,
					 "Failed to report completion: %d\n",
					 ret);
		}
	} else {
		ret_err = proverr2errno(event_rc);
		ret = cxip_cq_req_error(req, 0, ret_err, event_rc, NULL, 0,
					FI_ADDR_UNSPEC);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report error: %d\n", ret);
	}

	/* Only one outstanding request for triggered mode (notification PUT).
	 * The data PUT has event_success_disable=1, so it doesn't count
	 * toward outstanding requests.
	 */
	cxip_txc_otx_reqs_dec(req->rma.txc);
	cxip_evtq_req_free(req);

	return FI_SUCCESS;
}

static bool cxip_rma_emit_dma_need_req(size_t len, uint64_t flags,
				       struct cxip_mr *mr)
{
	/* DMA commands with FI_INJECT always require a request structure to
	 * track the bounce buffer.
	 */
	if (len && (flags & FI_INJECT))
		return true;

	/* If user request FI_COMPLETION, need request structure to return
	 * user context back.
	 *
	 * TODO: This can be optimized for zero byte operations. Specifically,
	 * The user context can be associated with the DMA command. But, this
	 * requires reworking on event queue processing to support.
	 */
	if (flags & FI_COMPLETION)
		return true;

	/* Writedata operations always require a request to track two-phase
	 * completion (data PUT + notification PUT).
	 */
	if (flags & FI_REMOTE_CQ_DATA)
		return true;

	/* If the user has provider their own MR, internal memory registration
	 * is not needed. Thus, no request structure is needed.
	 */
	if (mr)
		return false;

	/* In the initiator buffer length is zero, no memory registration is
	 * needed. Thus, no request structure is needed.
	 */
	if (!len)
		return false;

	return true;
}

/*
 * cxip_rma_writedata_select_mode() - Select writedata protocol mode.
 *
 * Determines which protocol to use for fi_writedata() based on message size,
 * MR type, and hardware capability.
 *
 * Mode selection (in priority order):
 *
 * 1. SINGLE_PUT: Small messages (≤ threshold) to standard MRs
 *    - One unrestricted PUT carries both data and header_data
 *    - Atomic operation, no race condition possible
 *    - Target data LE generates PUT event with immediate data
 *    - CSDG 4.2 (List Entry options): Standard MRs can generate full events
 *      for PUT operations. Optimized MRs have EVENT_COMM_DISABLE which
 *      "should not generate full events that indicate completion of a
 *      communication operation."
 *
 * 2. TRIGGERED: Large messages when triggered cmdq AND write_cntr available
 *    - Data PUT increments CT on ACK, has event_success_disable=1
 *    - Notification PUT triggered when CT reaches threshold
 *    - CSDG 5.4: "Issuance of the DMA is deferred until the success field of
 *      the counting event TRIG_CT reaches or exceeds the specific threshold"
 *    - Hardware-enforced sequencing eliminates race condition
 *    - Flush on data PUT ensures data visible before ACK (CSDG 5.1.2.3)
 *
 * 3. FLUSH: Fallback when triggered ops unavailable
 *    - Data PUT with flush, notification PUT emitted immediately after
 *    - CSDG 5.1.2.3: "If the FLUSH option is set... return of a response is
 *      delayed until the write is visible on the target node"
 *    - WARNING: Potential race condition - notification PUT (unrestricted)
 *      may arrive at target before data PUT (restricted/unordered) completes.
 *      CSDG 2.4.1: "Restricted commands are unordered"
 *      CSDG 5.6.4: "restricted commands may take any path through the network"
 *      The flush only delays the initiator-side ACK, not the notification.
 *    - Two events tracked in software
 *
 * @txc: Transmit context (for domain access)
 * @len: Message length
 * @key: Remote MR key
 * @return: Selected writedata mode
 */
/*
 * cxip_writedata_mode_str() - Return string name for writedata mode.
 */
static const char *cxip_writedata_mode_str(enum cxip_writedata_mode mode)
{
	switch (mode) {
	case CXIP_WRITEDATA_MODE_SINGLE_PUT:
		return "SINGLE_PUT";
	case CXIP_WRITEDATA_MODE_TRIGGERED:
		return "TRIGGERED";
	case CXIP_WRITEDATA_MODE_FLUSH:
		return "FLUSH";
	default:
		return "UNKNOWN";
	}
}

/*
 * cxip_writedata_stats_inc() - Increment writedata mode counter.
 *
 * Called under ep_obj->lock.
 */
static void cxip_writedata_stats_inc(struct cxip_txc *txc,
				     enum cxip_writedata_mode mode)
{
	switch (mode) {
	case CXIP_WRITEDATA_MODE_SINGLE_PUT:
		txc->writedata_stats.single_put_count++;
		break;
	case CXIP_WRITEDATA_MODE_TRIGGERED:
		txc->writedata_stats.triggered_count++;
		break;
	case CXIP_WRITEDATA_MODE_FLUSH:
		txc->writedata_stats.flush_count++;
		break;
	}
}

static enum cxip_writedata_mode
cxip_rma_writedata_select_mode(struct cxip_txc *txc, size_t len, uint64_t key)
{
	enum cxip_writedata_mode mode;
	bool is_opt_mr = cxip_generic_is_mr_key_opt(key);

	/* Single-PUT mode for ALL standard MR writedata operations.
	 *
	 * Standard MRs can receive header_data in PUT events on the data LE
	 * because they don't have EVENT_COMM_DISABLE set.
	 *
	 * CSDG 4.2 (LE Options): "EVENT_COMM_DISABLE - Specifies that this
	 * list entry should not generate full events that indicate completion
	 * of a communication operation."
	 *
	 * Optimized MRs have EVENT_COMM_DISABLE, so header_data from the PUT
	 * won't generate a target event - we must use two-PUT mode for them.
	 *
	 * IMPORTANT: We use SINGLE_PUT for ALL sizes with standard MRs (not
	 * just small messages) because the two-PUT protocol has a race
	 * condition. The data PUT and notification PUT go to different PTEs
	 * (ctrl.pte and writedata_pte), and the NIC may generate the
	 * notification event before the data PUT's DMA completes, even when
	 * both PUTs are unrestricted and arrive in order. This causes the
	 * application to see stale data when it reads the buffer after
	 * receiving the CQ completion.
	 *
	 * Since standard MRs already require unrestricted mode (due to
	 * match_bits addressing), there's no performance penalty for using
	 * SINGLE_PUT regardless of message size.
	 *
	 * The writedata_atomic_threshold env var is ignored for standard MRs
	 * but still applies to optimized MRs (where two-PUT is required).
	 */
	if (!is_opt_mr) {
		mode = CXIP_WRITEDATA_MODE_SINGLE_PUT;
		goto done;
	}

	/* Check if user wants to skip TRIGGERED mode and go straight to FLUSH.
	 * Useful for debugging or when triggered operations cause issues.
	 */
	if (cxip_env.writedata_prefer_flush) {
		mode = CXIP_WRITEDATA_MODE_FLUSH;
		goto done;
	}

	/* Triggered mode for optimized MRs (requires counter binding).
	 *
	 * Note: We only reach this code path for OPTIMIZED MRs since standard
	 * MRs always use SINGLE_PUT mode (see above).
	 *
	 * Requirements:
	 * 1. Domain has triggered cmdq available (trig_cmdq != NULL)
	 * 2. write_cntr is bound (we use it for CT sequencing)
	 *
	 * CSDG 5.4: "Triggered operations are queued in the initiating NIC
	 * until a threshold condition is met."
	 *
	 * The write_cntr's CT is incremented by the data PUT on ACK. The
	 * notification PUT is triggered when CT reaches the threshold.
	 *
	 * WARNING: Two-PUT mode has the same race condition as FLUSH mode.
	 * See the comment at FLUSH mode below.
	 */
	if (txc->domain->trig_cmdq != NULL && txc->write_cntr != NULL) {
		mode = CXIP_WRITEDATA_MODE_TRIGGERED;
		goto done;
	}

	/* Fallback to flush-based two-PUT protocol (optimized MRs only).
	 *
	 * Note: We only reach this code path for OPTIMIZED MRs since standard
	 * MRs always use SINGLE_PUT mode (see above). Optimized MRs require
	 * two-PUT because they have EVENT_COMM_DISABLE set on the data LE.
	 *
	 * FLUSH mode ordering:
	 * - Both data PUT and notification PUT are UNRESTRICTED
	 * - CSDG 5.6.4 guarantees ordered delivery to destination
	 * - Flush ensures data visible before ACK (CSDG 5.1.2.3)
	 *
	 * WARNING: Two-PUT mode has an inherent race condition for standard
	 * MRs because the data PUT and notification PUT go to different PTEs.
	 * The NIC may generate the notification event before data DMA
	 * completes. This is why standard MRs use SINGLE_PUT unconditionally.
	 * Optimized MRs accept this limitation since they cannot use
	 * SINGLE_PUT.
	 */
	mode = CXIP_WRITEDATA_MODE_FLUSH;

done:
	/* Debug logging for mode selection decisions */
	TXC_DBG(txc,
		"writedata mode=%s len=%zu threshold=%zu opt_mr=%d "
		"trig_cmdq=%d write_cntr=%d prefer_flush=%d\n",
		cxip_writedata_mode_str(mode), len,
		cxip_env.writedata_atomic_threshold, is_opt_mr,
		txc->domain->trig_cmdq != NULL, txc->write_cntr != NULL,
		cxip_env.writedata_prefer_flush);

	return mode;
}

/* Legacy helper for backward compatibility during transition */
static inline bool cxip_rma_writedata_use_single_put(struct cxip_txc *txc,
						     size_t len, uint64_t key)
{
	return cxip_rma_writedata_select_mode(txc, len, key) ==
	       CXIP_WRITEDATA_MODE_SINGLE_PUT;
}

static int cxip_rma_emit_dma(struct cxip_txc *txc, const void *buf, size_t len,
			     struct cxip_mr *mr, struct cxip_addr *caddr,
			     union c_fab_addr *dfa, uint8_t *idx_ext,
			     uint16_t vni, uint64_t addr, uint64_t key,
			     uint64_t data, uint64_t flags, void *context,
			     bool write, bool unr, uint32_t tclass,
			     enum cxi_traffic_class_type tc_type,
			     bool triggered, uint64_t trig_thresh,
			     struct cxip_cntr *trig_cntr,
			     struct cxip_cntr *comp_cntr)
{
	struct cxip_req *req = NULL;
	struct cxip_md *dma_md = NULL;
	void *dma_buf = NULL;
	struct c_full_dma_cmd dma_cmd = {};
	int ret;
	struct cxip_domain *dom = txc->domain;
	struct cxip_cntr *cntr;
	void *inject_req;
	uint64_t access = write ? CXI_MAP_READ : CXI_MAP_WRITE;

	/* MR desc cannot be value unless hybrid MR desc is enabled. */
	if (!dom->hybrid_mr_desc)
		mr = NULL;

	if (cxip_rma_emit_dma_need_req(len, flags, mr)) {
		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc, "Failed to allocate request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err;
		}

		req->context = (uint64_t) context;
		req->flags = FI_RMA | (write ? FI_WRITE : FI_READ) |
			     (flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
		req->trig_cntr = trig_cntr;

		/* Writedata operations: select protocol mode based on message
		 * size, MR type, and hardware capability.
		 *
		 * SINGLE_PUT mode (small messages to standard MRs):
		 * - One unrestricted PUT carries data + header_data
		 * - Target data LE generates PUT event with header_data
		 * - Single completion event expected, standard callback
		 *
		 * TRIGGERED mode (large messages, hardware-sequenced):
		 * - Data PUT increments CT on ACK, suppresses event
		 * - Notification PUT triggered when CT threshold reached
		 * - Single event expected (from notification PUT)
		 * - Hardware-enforced sequencing eliminates race (CSDG 5.4)
		 *
		 * FLUSH mode (fallback):
		 * - Data PUT with flush + notification PUT
		 * - Two events tracked in software
		 */
		if (flags & FI_REMOTE_CQ_DATA) {
			enum cxip_writedata_mode mode =
				cxip_rma_writedata_select_mode(txc, len, key);

			req->rma.writedata_mode = mode;

			switch (mode) {
			case CXIP_WRITEDATA_MODE_SINGLE_PUT:
				/* Standard callback, single event */
				req->cb = cxip_rma_cb;
				req->rma.writedata = false;
				break;

			case CXIP_WRITEDATA_MODE_TRIGGERED:
				/* Triggered callback, single event from
				 * notification PUT. Data PUT event suppressed.
				 */
				req->cb = cxip_rma_writedata_triggered_cb;
				req->rma.writedata = false;
				req->rma.writedata_trig_cntr = txc->write_cntr;
				break;

			case CXIP_WRITEDATA_MODE_FLUSH:
				/* Flush callback, two events tracked */
				req->cb = cxip_rma_writedata_cb;
				req->rma.writedata = true;
				req->rma.writedata_events = 0;
				req->rma.writedata_events_expected = 2;
				req->rma.writedata_data_rc = C_RC_OK;
				req->rma.writedata_notify_rc = C_RC_OK;
				break;
			}

			/* Increment stats now that mode is committed to req */
			cxip_writedata_stats_inc(txc, mode);
		} else {
			req->cb = cxip_rma_cb;
			req->rma.writedata = false;
		}
	}

	if (len) {
		/* If the operation is an DMA inject operation (which can occur
		 * when doing RMA commands to unoptimized MRs), a provider
		 * bounce buffer is always needed to store the user payload.
		 *
		 * Always prefer user-provided MR over internally mapping the
		 * buffer.
		 */
		if (flags & FI_INJECT) {
			assert(req != NULL);

			req->rma.ibuf = cxip_txc_ibuf_alloc(txc);
			if (!req->rma.ibuf) {
				ret = -FI_EAGAIN;
				TXC_WARN(txc,
					 "Failed to allocate bounce buffer: "
					 "%d:%s\n",
					 ret, fi_strerror(-ret));
				goto err_free_cq_req;
			}

			ret = cxip_txc_copy_from_hmem(txc, NULL, req->rma.ibuf,
						      buf, len);
			if (ret) {
				TXC_WARN(txc,
					 "cxip_txc_copy_from_hmem failed: "
					 "%d:%s\n",
					 ret, fi_strerror(-ret));
				goto err_free_rma_buf;
			}

			dma_buf = (void *) req->rma.ibuf;
			dma_md = cxip_txc_ibuf_md(req->rma.ibuf);
		} else if (mr) {
			dma_buf = (void *) buf;
			dma_md = mr->md;
		} else {
			assert(req != NULL);

			ret = cxip_ep_obj_map(txc->ep_obj, buf, len, access, 0,
					      &req->rma.local_md);
			if (ret) {
				TXC_WARN(txc, "Failed to map buffer: %d:%s\n",
					 ret, fi_strerror(-ret));
				goto err_free_cq_req;
			}

			dma_buf = (void *) buf;
			dma_md = req->rma.local_md;
		}
	}

	dma_cmd.command.cmd_type = C_CMD_TYPE_DMA;
	dma_cmd.index_ext = *idx_ext;
	dma_cmd.event_send_disable = 1;
	dma_cmd.dfa = *dfa;
	ret = cxip_adjust_remote_offset(&addr, key);
	if (ret) {
		TXC_WARN(txc, "Remote offset overflow\n");
		goto err_free_cq_req;
	}
	dma_cmd.remote_offset = addr;
	dma_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);
	dma_cmd.match_bits = CXIP_KEY_MATCH_BITS(key);

	/* For single-PUT writedata operations, set the cq_data bit in
	 * match_bits so the target MR can distinguish this from regular writes
	 * and generate a CQ completion with FI_REMOTE_CQ_DATA.
	 *
	 * For two-PUT modes (TRIGGERED, FLUSH), do NOT set this bit on the
	 * data PUT. The CQ completion with immediate data comes from the
	 * notification PUT to the separate notification LE, not from the data
	 * PUT. Setting this bit on the data PUT would cause a spurious CQ
	 * completion with header_data=0.
	 */
	if ((flags & FI_REMOTE_CQ_DATA) &&
	    cxip_rma_writedata_use_single_put(txc, len, key))
		dma_cmd.match_bits |= CXIP_MR_KEY_CQ_DATA_BIT;

	if (req) {
		dma_cmd.user_ptr = (uint64_t) req;
	} else {
		if (write)
			inject_req =
				cxip_rma_write_selective_completion_req(txc);
		else
			inject_req =
				cxip_rma_read_selective_completion_req(txc);

		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_rma_buf;
		}

		dma_cmd.user_ptr = (uint64_t) inject_req;
		dma_cmd.event_success_disable = 1;
	}

	if (!unr)
		dma_cmd.restricted = 1;

	if (write) {
		dma_cmd.command.opcode = C_CMD_PUT;

		/* Writedata handling depends on single-PUT vs two-PUT mode:
		 *
		 * Single-PUT mode (small standard MR):
		 * - This single PUT carries both data and header_data
		 * - Target data LE generates PUT event with header_data
		 * - No notification PUT needed
		 *
		 * Two-PUT mode (large or optimized MR):
		 * - This PUT carries only data (restricted, no header_data)
		 * - Separate notification PUT carries header_data
		 * - Use flush for ordering (CSDG 5.1.2.3)
		 */

		/* Triggered DMA operations have their own completion counter
		 * and the one associated with the TXC cannot be used.
		 */
		cntr = triggered ? comp_cntr : txc->write_cntr;
		if (cntr) {
			dma_cmd.event_ct_ack = 1;
			dma_cmd.ct = cntr->ct->ctn;

			/* For triggered writedata mode, we DON'T track
			 * progress on the data PUT's counter here. The data
			 * PUT has event_success_disable=1 (set below), so it
			 * won't generate an event. Instead, we track progress
			 * on writedata_trig_cntr when emitting the triggered
			 * notification PUT, and decrement in the triggered
			 * callback.
			 *
			 * For other modes (SINGLE_PUT, FLUSH), we track
			 * progress normally since the data PUT generates an
			 * event.
			 */
			if (req && !((flags & FI_REMOTE_CQ_DATA) &&
				     req->rma.writedata_mode ==
					     CXIP_WRITEDATA_MODE_TRIGGERED)) {
				req->rma.cntr = cntr;
				cxip_cntr_progress_inc(cntr);
			}
		}

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			dma_cmd.flush = 1;

		/* For writedata in two-PUT mode, enable flush on the data PUT.
		 *
		 * CSDG References for ordering:
		 *
		 * CSDG 5.1.2.3: "If the FLUSH option is set... return of a
		 *   response is delayed until the write is visible on the
		 *   target node"
		 * CSDG 5.6.3: "For each queue, the commands are issued in
		 *   order onto the network."
		 * CSDG 5.6.4: "Messages injected into the network resulting
		 *   from unrestricted commands will be delivered to the
		 *   destination Cassini in order."
		 *
		 * Mode-specific ordering:
		 *
		 * FLUSH mode: Both data PUT and notification PUT are
		 *   UNRESTRICTED (cxip_rma_is_unrestricted returns true).
		 *   This ensures they travel the same ordered network path.
		 *   Combined with flush, the notification arrives after data
		 *   is visible in target memory.
		 *
		 * TRIGGERED mode: Data PUT can be RESTRICTED for better
		 *   performance. Ordering is enforced by hardware CT
		 *   thresholds - notification PUT is triggered only after
		 *   data PUT ACK arrives (CSDG 5.4). Flush ensures data
		 *   visibility before ACK.
		 *
		 * SINGLE_PUT mode: Only one operation, no ordering concern.
		 */
		if (flags & FI_REMOTE_CQ_DATA)
			dma_cmd.flush = 1;

		/* For TRIGGERED writedata mode, suppress the data PUT's
		 * completion event. The notification PUT (triggered op) will
		 * be the only event we receive.
		 *
		 * The data PUT still increments the CT on ACK (event_ct_ack=1
		 * set above), which triggers the notification PUT.
		 */
		if ((flags & FI_REMOTE_CQ_DATA) && req &&
		    req->rma.writedata_mode == CXIP_WRITEDATA_MODE_TRIGGERED) {
			dma_cmd.event_success_disable = 1;
		}

		/* For single-PUT writedata, add header_data to this PUT.
		 * The unrestricted PUT will deliver it to the target data LE.
		 */
		if ((flags & FI_REMOTE_CQ_DATA) &&
		    cxip_rma_writedata_use_single_put(txc, len, key)) {
			dma_cmd.header_data = data;
		}
	} else {
		dma_cmd.command.opcode = C_CMD_GET;

		/* Triggered DMA operations have their own completion counter
		 * and the one associated with the TXC cannot be used.
		 */
		cntr = triggered ? comp_cntr : txc->read_cntr;
		if (cntr) {
			dma_cmd.event_ct_reply = 1;
			dma_cmd.ct = cntr->ct->ctn;

			if (req) {
				req->rma.cntr = cntr;
				cxip_cntr_progress_inc(cntr);
			}
		}
	}

	/* Only need to fill if DMA command address fields if MD is valid. */
	if (dma_md) {
		dma_cmd.lac = dma_md->md->lac;
		dma_cmd.local_addr = CXI_VA_TO_IOVA(dma_md->md, dma_buf);
		dma_cmd.request_len = len;
	}

	ret = cxip_txc_emit_dma(txc, vni, cxip_ofi_to_cxi_tc(tclass), tc_type,
				trig_cntr, trig_thresh, &dma_cmd, flags);
	if (ret) {
		TXC_WARN(txc, "Failed to emit dma command: %d:%s\n", ret,
			 fi_strerror(-ret));
		goto err_free_rma_buf;
	}

	/* For writedata operations using two-PUT protocol (TRIGGERED or FLUSH),
	 * emit a 0-length unrestricted PUT to the notification PTE. This PUT
	 * carries the immediate data (header_data) and the data length
	 * (remote_offset) to the target's notification LE.
	 *
	 * This is SKIPPED for SINGLE_PUT mode, where the data PUT itself
	 * carries header_data directly to the target's data LE.
	 *
	 * The notification PUT is:
	 * - Unrestricted (to deliver header_data)
	 * - 0-length (no actual data, just signaling)
	 * - Targeted at CXIP_PTL_IDX_WRITEDATA_NOTIFY PTE
	 * - match_bits = key (to match the correct MR's notification LE)
	 * - remote_offset = len (to convey the actual data length)
	 * - header_data = data (the immediate data from fi_writedata)
	 *
	 * Mode-specific behavior:
	 *
	 * TRIGGERED mode:
	 * - Data PUT has event_success_disable=1 (no completion event)
	 * - Data PUT increments write_cntr CT on ACK
	 * - Notification PUT is queued as triggered op with threshold
	 * - Hardware fires notification only after data PUT ACK (CSDG 5.4)
	 * - Single event from notification PUT
	 *
	 * FLUSH mode:
	 * - Data PUT has flush=1 for ordering (CSDG 5.1.2.3)
	 * - Both PUTs generate completion events
	 * - Callback tracks both completions
	 */
	if ((flags & FI_REMOTE_CQ_DATA) && write &&
	    !cxip_rma_writedata_use_single_put(txc, len, key)) {
		struct c_full_dma_cmd notify_cmd = {};
		union c_fab_addr notify_dfa;
		uint8_t notify_idx_ext;
		enum cxip_writedata_mode mode = req->rma.writedata_mode;

		assert(req != NULL); /* writedata always allocates a request */

		/* Build DFA for notification PTE using same target address */
		cxi_build_dfa(caddr->nic, caddr->pid, txc->pid_bits,
			      CXIP_PTL_IDX_WRITEDATA_NOTIFY, &notify_dfa,
			      &notify_idx_ext);

		notify_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		notify_cmd.command.opcode = C_CMD_PUT;
		notify_cmd.index_ext = notify_idx_ext;
		notify_cmd.dfa = notify_dfa;
		notify_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);

		/* match_bits = key to find the correct notification LE */
		notify_cmd.match_bits = CXIP_KEY_MATCH_BITS(key);

		/* remote_offset encodes the data length for the target.
		 * With MANAGE_LOCAL on the LE, HW ignores this for buffer
		 * addressing, but delivers it in the event for SW to use.
		 */
		notify_cmd.remote_offset = len;

		/* header_data carries the immediate data from fi_writedata */
		notify_cmd.header_data = data;

		/* 0-length PUT - no local buffer needed */
		notify_cmd.request_len = 0;

		/* Unrestricted to deliver header_data */
		notify_cmd.restricted = 0;

		/* Common: disable send events, point to request */
		notify_cmd.event_send_disable = 1;
		notify_cmd.user_ptr = (uint64_t) req;

		if (mode == CXIP_WRITEDATA_MODE_TRIGGERED) {
			/* TRIGGERED mode: emit as triggered operation.
			 *
			 * CSDG 5.4: "Issuance of the DMA is deferred until the
			 * success field of the counting event TRIG_CT reaches
			 * or exceeds the specific threshold value"
			 *
			 * CSDG 5.4: "If a counting event has already reached
			 * the threshold value when a triggered operation is
			 * processed, that operation is processed immediately."
			 * (This is fine - CT at threshold means data PUT ACK
			 * already arrived, so data is visible due to flush.)
			 *
			 * The data PUT (already emitted above) has:
			 * - event_ct_ack=1: increments CT on ACK (CSDG 5.3)
			 * - event_success_disable=1: no completion event
			 * - flush=1: ACK delayed until data visible
			 *
			 * The notification PUT:
			 * - event_success_disable=0: generates completion event
			 * - Triggered when CT reaches threshold
			 */
			uint64_t trig_threshold;
			struct c_ct_writeback wb;

			notify_cmd.event_success_disable = 0;

			/* Get current counter value and set threshold.
			 *
			 * CSDG 5.3.2: "When [write-back] is enabled, the value
			 * of a counting event is written back to client memory"
			 *
			 * We read the current CT value and set threshold to
			 * current + 1. The data PUT will increment CT on ACK.
			 * The triggered notification fires when CT >=
			 * threshold.
			 */
			ret = cxip_cntr_get_wb(req->rma.writedata_trig_cntr,
					       &wb);
			if (ret) {
				TXC_WARN(txc,
					 "Failed to read CT value: %d:%s\n",
					 ret, fi_strerror(-ret));
				/* Data PUT already sent with event suppressed.
				 * No event will arrive, so we must complete
				 * the request synchronously with an error.
				 */
				goto triggered_emit_failed;
			}
			trig_threshold = wb.ct_success + 1;
			req->rma.writedata_trig_thresh = trig_threshold;

			/* Track progress on the triggered counter */
			cxip_cntr_progress_inc(req->rma.writedata_trig_cntr);

			/* Use CXI_TC_TYPE_DEFAULT: the notification PUT is
			 * always a 0-length unrestricted PUT (just carries
			 * header_data), and the triggered command queue only
			 * supports DEFAULT tc_type. FI_REMOTE_CQ_DATA is
			 * stripped since we've already encoded it in the
			 * header_data field.
			 */
			ret = cxip_txc_emit_dma(
				txc, vni, cxip_ofi_to_cxi_tc(tclass),
				CXI_TC_TYPE_DEFAULT,
				req->rma.writedata_trig_cntr, trig_threshold,
				&notify_cmd, flags & ~FI_REMOTE_CQ_DATA);
			if (ret) {
				TXC_WARN(txc,
					 "Failed to emit triggered notify: "
					 "%d:%s\n",
					 ret, fi_strerror(-ret));
				/* Data PUT already sent with event suppressed.
				 * No event will arrive, so we must complete
				 * the request synchronously with an error.
				 * Undo the progress increment we just did.
				 */
				cxip_cntr_progress_dec(
					req->rma.writedata_trig_cntr);
				goto triggered_emit_failed;
			}
		} else {
			/* FLUSH mode: emit notification immediately.
			 *
			 * Both data PUT and notification PUT generate events.
			 * Flush on data PUT ensures ordering.
			 * Callback tracks both completions.
			 */
			notify_cmd.event_success_disable = 0;

			ret = cxip_txc_emit_dma(
				txc, vni, cxip_ofi_to_cxi_tc(tclass),
				CXI_TC_TYPE_DEFAULT, NULL, 0, &notify_cmd,
				flags & ~FI_REMOTE_CQ_DATA);
			if (ret) {
				TXC_WARN(txc,
					 "Failed to emit notify PUT: %d:%s\n",
					 ret, fi_strerror(-ret));
				/* Data PUT already sent - can't rollback.
				 * Reduce expected events so callback completes
				 * after just the data PUT.
				 */
				req->rma.writedata_events_expected = 1;
				req->rma.writedata_notify_rc = C_RC_NO_SPACE;
				return FI_SUCCESS;
			}
		}
	}

	return FI_SUCCESS;

triggered_emit_failed:
	/* Triggered mode emit failed after the data PUT was sent with
	 * event_success_disable=1. No completion event will arrive for the
	 * data PUT, so we must complete the request synchronously here.
	 *
	 * The data PUT was already sent successfully, so the data transfer
	 * itself may succeed at the target. But we couldn't send the
	 * notification, so report an error to the user.
	 *
	 * Clean up resources and report error via CQ.
	 */
	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	if (req->rma.ibuf)
		cxip_txc_ibuf_free(txc, req->rma.ibuf);

	/* Report error - use ENOMEM since emit failure is resource exhaustion
	 */
	cxip_cq_req_error(req, 0, FI_ENOMEM, C_RC_NO_SPACE, NULL, 0,
			  FI_ADDR_UNSPEC);

	cxip_evtq_req_free(req);
	return FI_SUCCESS;

err_free_rma_buf:
	if (req && req->rma.ibuf)
		cxip_txc_ibuf_free(txc, req->rma.ibuf);
err_free_cq_req:
	if (req)
		cxip_evtq_req_free(req);
err:
	return ret;
}

static int cxip_rma_emit_idc(struct cxip_txc *txc, const void *buf, size_t len,
			     struct cxip_addr *caddr, union c_fab_addr *dfa,
			     uint8_t *idx_ext, uint16_t vni, uint64_t addr,
			     uint64_t key, uint64_t data, uint64_t flags,
			     void *context, bool unr, uint32_t tclass,
			     enum cxi_traffic_class_type tc_type)
{
	int ret;
	struct cxip_req *req = NULL;
	void *hmem_buf = NULL;
	void *idc_buf;
	struct c_cstate_cmd cstate_cmd = {};
	struct c_idc_put_cmd idc_put = {};
	void *inject_req;

	/* IDCs must be tracked if the user requests a completion event,
	 * or for writedata which needs two-phase completion tracking.
	 */
	if ((flags & FI_COMPLETION) || (flags & FI_REMOTE_CQ_DATA)) {
		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc, "Failed to allocate request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err;
		}

		req->context = (uint64_t) context;
		req->flags = FI_RMA | FI_WRITE | (flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;

		/* Writedata operations: select protocol mode.
		 *
		 * Note: IDC is only used for optimized MRs (see
		 * cxip_rma_is_idc), and SINGLE_PUT mode requires standard MRs,
		 * so SINGLE_PUT will never apply here. Also, triggered
		 * operations never use IDC (see cxip_rma_is_idc), so
		 * TRIGGERED mode won't apply either. IDC writedata always
		 * uses FLUSH mode.
		 *
		 * The check is included for consistency and future-proofing.
		 */
		if (flags & FI_REMOTE_CQ_DATA) {
			enum cxip_writedata_mode mode =
				cxip_rma_writedata_select_mode(txc, len, key);

			req->rma.writedata_mode = mode;

			/* IDC can only use FLUSH mode - SINGLE_PUT requires
			 * standard MRs (IDC uses optimized), and TRIGGERED
			 * requires DMA commands (IDC uses IDC commands).
			 * Force to FLUSH if somehow another mode was selected.
			 */
			if (mode != CXIP_WRITEDATA_MODE_SINGLE_PUT) {
				mode = CXIP_WRITEDATA_MODE_FLUSH;
				req->rma.writedata_mode = mode;
			}

			switch (mode) {
			case CXIP_WRITEDATA_MODE_SINGLE_PUT:
				/* Never reached for IDC */
				req->cb = cxip_rma_cb;
				req->rma.writedata = false;
				break;

			case CXIP_WRITEDATA_MODE_TRIGGERED:
				/* Never reached for IDC - fall through to FLUSH
				 */
			case CXIP_WRITEDATA_MODE_FLUSH:
				/* Flush mode: two events tracked */
				req->cb = cxip_rma_writedata_cb;
				req->rma.writedata = true;
				req->rma.writedata_events = 0;
				req->rma.writedata_events_expected = 2;
				req->rma.writedata_data_rc = C_RC_OK;
				req->rma.writedata_notify_rc = C_RC_OK;
				break;
			}

			/* Increment stats now that mode is committed to req */
			cxip_writedata_stats_inc(txc, mode);
		} else {
			req->cb = cxip_rma_cb;
			req->rma.writedata = false;
		}
	}

	/* If HMEM is request and since the buffer type may not be host memory,
	 * doing a memcpy could result in a segfault. Thus, an HMEM bounce
	 * buffer is required to ensure IDC payload is in host memory.
	 */
	if (txc->hmem && len) {
		hmem_buf = cxip_txc_ibuf_alloc(txc);
		if (!hmem_buf) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate bounce buffer: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_cq_req;
		}

		ret = cxip_txc_copy_from_hmem(txc, NULL, hmem_buf, buf, len);
		if (ret) {
			TXC_WARN(txc, "cxip_txc_copy_from_hmem failed: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		idc_buf = hmem_buf;
	} else {
		idc_buf = (void *) buf;
	}

	cstate_cmd.event_send_disable = 1;
	cstate_cmd.index_ext = *idx_ext;
	cstate_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);

	if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
		cstate_cmd.flush = 1;

	/* For writedata operations, always use flush to ensure data
	 * is visible at the target before the notification PUT arrives.
	 * Flush delays ACK until write visible (CSDG 5.1.2.3). Required
	 * because restricted PUTs are unordered in network (CSDG 5.6.4).
	 */
	if (flags & FI_REMOTE_CQ_DATA)
		cstate_cmd.flush = 1;

	if (!unr)
		cstate_cmd.restricted = 1;

	if (txc->write_cntr) {
		cstate_cmd.event_ct_ack = 1;
		cstate_cmd.ct = txc->write_cntr->ct->ctn;

		if (req) {
			req->rma.cntr = txc->write_cntr;
			cxip_cntr_progress_inc(txc->write_cntr);
		}
	}

	/* If the user has not request a completion, success events will be
	 * disabled. But, if for some reason the operation completes with an
	 * error, an event will occur. For this case, a TXC inject request is
	 * allocated. This request enables the reporting of failed operation to
	 *  the completion queue. This request is freed when the TXC is closed.
	 */
	if (req) {
		cstate_cmd.user_ptr = (uint64_t) req;
	} else {
		inject_req = cxip_rma_write_selective_completion_req(txc);
		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		cstate_cmd.user_ptr = (uint64_t) inject_req;
		cstate_cmd.event_success_disable = 1;
	}

	idc_put.idc_header.dfa = *dfa;

	ret = cxip_adjust_remote_offset(&addr, key);
	if (ret) {
		TXC_WARN(txc, "Remote offset overflow\n");
		goto err_free_hmem_buf;
	}
	idc_put.idc_header.remote_offset = addr;

	ret = cxip_txc_emit_idc_put(txc, vni, cxip_ofi_to_cxi_tc(tclass),
				    tc_type, &cstate_cmd, &idc_put, idc_buf,
				    len, flags);
	if (ret) {
		TXC_WARN(txc, "Failed to emit idc_put command: %d:%s\n", ret,
			 fi_strerror(-ret));
		goto err_free_hmem_buf;
	}

	if (hmem_buf)
		cxip_txc_ibuf_free(txc, hmem_buf);

	/* For writedata operations using two-PUT protocol, emit a 0-length
	 * unrestricted DMA PUT to the notification PTE. IDC commands don't
	 * support header_data, so we must use a DMA command for the
	 * notification PUT even when the data transfer uses IDC.
	 *
	 * This is SKIPPED for single-PUT mode (which can't apply to IDC anyway,
	 * since IDC requires optimized MRs and single-PUT requires standard
	 * MRs).
	 *
	 * Same critical fix as in cxip_rma_emit_dma: we must enable events
	 * on the notification PUT (event_success_disable=0) so CMD_CQ_FENCE
	 * can track both commands (CSDG 5.6.3, 5.1.2.3).
	 */
	if ((flags & FI_REMOTE_CQ_DATA) &&
	    !cxip_rma_writedata_use_single_put(txc, len, key)) {
		struct c_full_dma_cmd notify_cmd = {};
		union c_fab_addr notify_dfa;
		uint8_t notify_idx_ext;

		assert(req != NULL); /* writedata always allocates a request */

		/* Build DFA for notification PTE using same target address */
		cxi_build_dfa(caddr->nic, caddr->pid, txc->pid_bits,
			      CXIP_PTL_IDX_WRITEDATA_NOTIFY, &notify_dfa,
			      &notify_idx_ext);

		notify_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		notify_cmd.command.opcode = C_CMD_PUT;
		notify_cmd.index_ext = notify_idx_ext;
		notify_cmd.dfa = notify_dfa;
		notify_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);

		/* match_bits = key to find the correct notification LE */
		notify_cmd.match_bits = CXIP_KEY_MATCH_BITS(key);

		/* remote_offset encodes the data length for the target */
		notify_cmd.remote_offset = len;

		/* header_data carries the immediate data from fi_writedata */
		notify_cmd.header_data = data;

		/* 0-length PUT - no local buffer needed */
		notify_cmd.request_len = 0;

		/* Unrestricted to deliver header_data */
		notify_cmd.restricted = 0;

		/* CRITICAL: Enable completion events for fence semantics.
		 * Point to the same request as the IDC PUT so the callback
		 * can track both completions. CMD_CQ_FENCE waits for
		 * completions (CSDG 5.6.3); event_success_disable=0 required
		 * (CSDG 5.1.2.3).
		 */
		notify_cmd.event_send_disable = 1;
		notify_cmd.event_success_disable = 0;
		notify_cmd.user_ptr = (uint64_t) req;

		ret = cxip_txc_emit_dma(txc, vni, cxip_ofi_to_cxi_tc(tclass),
					CXI_TC_TYPE_DEFAULT, NULL, 0,
					&notify_cmd, flags);
		if (ret) {
			TXC_WARN(txc,
				 "Failed to emit notify PUT for IDC: %d:%s\n",
				 ret, fi_strerror(-ret));
			/* IDC PUT already sent - can't rollback. Reduce
			 * expected events so callback completes after just the
			 * IDC PUT. Store error so callback reports it to user.
			 * The target will get the data but not the completion
			 * notification. Use C_RC_NO_SPACE since emit failures
			 * are typically due to command queue resource
			 * exhaustion.
			 */
			req->rma.writedata_events_expected = 1;
			req->rma.writedata_notify_rc = C_RC_NO_SPACE;
			return FI_SUCCESS;
		}
	}

	return FI_SUCCESS;

err_free_hmem_buf:
	if (hmem_buf)
		cxip_txc_ibuf_free(txc, hmem_buf);
err_free_cq_req:
	if (req)
		cxip_evtq_req_free(req);
err:
	return ret;
}

static bool cxip_rma_is_unrestricted(struct cxip_txc *txc, uint64_t key,
				     uint64_t msg_order, bool write,
				     uint64_t flags, size_t len)
{
	/* Unoptimized keys are implemented with match bits and must always be
	 * unrestricted.
	 */
	if (!cxip_generic_is_mr_key_opt(key))
		return true;

	/* Writedata ordering analysis for two-PUT protocol modes:
	 *
	 * CSDG 2.4.1: "Restricted commands are unordered."
	 * CSDG 5.6.4: "restricted commands may take any path through the
	 *   network to the destination"
	 * CSDG 5.6.4: "Messages injected into the network resulting from
	 *   unrestricted commands will be delivered to the destination Cassini
	 *   in order."
	 *
	 * For SINGLE_PUT mode: Only one operation, no ordering concern.
	 * Returns false here, allowing the single PUT to be restricted
	 * (it will be made unrestricted elsewhere due to FI_RMA_EVENT).
	 *
	 * For TRIGGERED mode: Hardware CT-based sequencing ensures the
	 * notification PUT is not sent until the data PUT completes. The
	 * data PUT can be restricted for optimal performance - ordering
	 * is enforced by the trigger threshold mechanism (CSDG 5.4).
	 *
	 * For FLUSH mode: Both data PUT and notification PUT are sent
	 * immediately. The flush option only delays the initiator-side ACK
	 * (CSDG 5.1.2.3), it does NOT prevent the notification PUT from
	 * arriving at the target before the data PUT.
	 *
	 * To fix the FLUSH mode race condition, BOTH the data PUT and
	 * notification PUT must be unrestricted. This ensures they travel
	 * through the same ordered network path, and since the data PUT
	 * is emitted first (CSDG 5.6.3: "commands are issued in order onto
	 * the network"), the notification will arrive after the data.
	 */
	if (flags & FI_REMOTE_CQ_DATA) {
		enum cxip_writedata_mode mode =
			cxip_rma_writedata_select_mode(txc, len, key);

		switch (mode) {
		case CXIP_WRITEDATA_MODE_SINGLE_PUT:
			/* Single PUT to standard MR - will be unrestricted
			 * due to FI_RMA_EVENT check below. Return false here
			 * to let that logic handle it.
			 */
			return false;

		case CXIP_WRITEDATA_MODE_TRIGGERED:
			/* Hardware-sequenced: data PUT can be restricted,
			 * notification is triggered after data PUT ACK.
			 */
			return false;

		case CXIP_WRITEDATA_MODE_FLUSH:
			/* FLUSH mode requires unrestricted data PUT to
			 * guarantee ordering with the notification PUT.
			 */
			return true;
		}
	}

	/* If MR indicates remote events are required unrestricted must be
	 * used. If the MR is a client key, we assume if FI_RMA_EVENTS are
	 * requested, the remote client key MR is attached to a counter or
	 * requires RMA events, so unrestricted is used.
	 */
	if (cxip_generic_is_mr_key_events(txc->ep_obj->caps, key))
		return true;

	/* If the operation is an RMA write and the user has requested fabric
	 * write after write ordering, unrestricted must be used.
	 */
	if (write && msg_order & (FI_ORDER_WAW | FI_ORDER_RMA_WAW))
		return true;

	return false;
}

static bool cxip_rma_is_idc(struct cxip_txc *txc, uint64_t key, size_t len,
			    bool write, bool triggered, bool unr,
			    uint64_t flags)
{
	size_t max_idc_size = unr ? CXIP_INJECT_SIZE : C_MAX_IDC_PAYLOAD_RES;

	/* IDC commands are not supported for unoptimized MR since the IDC
	 * small message format does not support remote offset which is needed
	 * for RMA commands.
	 */
	if (!cxip_generic_is_mr_key_opt(key))
		return false;

	/* IDC commands are only support with RMA writes. */
	if (!write)
		return false;

	/* IDC commands only support a limited payload size. */
	if (len > max_idc_size)
		return false;

	/* Triggered operations never can be issued with an IDC. */
	if (triggered)
		return false;

	/* Don't issue non-inject operation as IDC if disabled by env */
	if (!(flags & FI_INJECT) && cxip_env.disable_non_inject_rma_idc)
		return false;

	/* IDC PUT with writedata is supported - the IDC handles the data
	 * transfer and a separate 0-length DMA PUT sends the notification.
	 */

	return true;
}

/*
 * cxip_rma_common() - Perform an RMA operation.
 *
 * Common RMA function. Performs RMA reads and writes of all kinds.
 *
 * Generally, operations are supported by Cassini DMA commands. IDC commands
 * are used instead for Write operations smaller than the maximum IDC payload
 * size.
 *
 * If the FI_COMPLETION flag is specified, the operation will generate a
 * libfabric completion event. If an event is not requested and an IDC command
 * is used, hardware success events will be suppressed. If a completion is
 * required but an IDC can't be used, the provider tracks the request
 * internally, but will suppress the libfabric event. The provider must track
 * DMA commands in order to clean up the source buffer mapping on completion.
 */
ssize_t cxip_rma_common(enum fi_op_type op, struct cxip_txc *txc,
			const void *buf, size_t len, void *desc,
			fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			uint64_t data, uint64_t flags, uint32_t tclass,
			uint64_t msg_order, void *context, bool triggered,
			uint64_t trig_thresh, struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr)
{
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_idx;
	enum cxi_traffic_class_type tc_type;
	bool write = op == FI_OP_WRITE;
	bool unr;
	bool idc;
	int ret;
	uint16_t vni;

	if (len && !buf) {
		TXC_WARN(txc, "Invalid buffer\n");
		return -FI_EINVAL;
	}

	if ((flags & FI_INJECT) && len > CXIP_INJECT_SIZE) {
		TXC_WARN(txc, "RMA inject size exceeds limit\n");
		return -FI_EMSGSIZE;
	}

	if (len > CXIP_EP_MAX_MSG_SZ) {
		TXC_WARN(txc, "RMA length exceeds limit\n");
		return -FI_EMSGSIZE;
	}

	if (!cxip_generic_is_valid_mr_key(key)) {
		TXC_WARN(txc, "Invalid remote key: 0x%lx\n", key);
		return -FI_EKEYREJECTED;
	}

	/* Writedata requires provider keys so we can verify the target MR
	 * has success events enabled (indicated by the 'events' bit in the
	 * key). Client keys don't encode this information, so we can't validate
	 * them.
	 */
	if (flags & FI_REMOTE_CQ_DATA) {
		struct cxip_mr_key mr_key = {.raw = key};

		if (!mr_key.is_prov) {
			TXC_WARN(txc,
				 "FI_REMOTE_CQ_DATA requires provider key "
				 "(FI_MR_PROV_KEY): key=0x%lx\n",
				 key);
			return -FI_EINVAL;
		}
		if (!cxip_generic_is_mr_key_events(txc->ep_obj->caps, key)) {
			TXC_WARN(txc,
				 "FI_REMOTE_CQ_DATA requires target MR with "
				 "FI_RMA_EVENT: key=0x%lx\n",
				 key);
			return -FI_EINVAL;
		}
	}

	unr = cxip_rma_is_unrestricted(txc, key, msg_order, write, flags, len);
	idc = cxip_rma_is_idc(txc, key, len, write, triggered, unr, flags);

	/* Build target network address. */
	ret = cxip_av_lookup_addr(txc->ep_obj->av, tgt_addr, &caddr);
	if (ret) {
		TXC_WARN(txc, "Failed to look up FI addr: %d:%s\n", ret,
			 fi_strerror(-ret));
		return ret;
	}

	if (txc->ep_obj->av_auth_key)
		vni = caddr.vni;
	else
		vni = txc->ep_obj->auth_key.vni;

	pid_idx = cxip_generic_mr_key_to_ptl_idx(txc->domain, key, write);
	cxi_build_dfa(caddr.nic, caddr.pid, txc->pid_bits, pid_idx, &dfa,
		      &idx_ext);

	/* Select the correct traffic class type within a traffic class. */
	if (!unr && (flags & FI_CXI_HRP))
		tc_type = CXI_TC_TYPE_HRP;
	else if (!unr && !triggered)
		tc_type = CXI_TC_TYPE_RESTRICTED;
	else
		tc_type = CXI_TC_TYPE_DEFAULT;

	/* IDC commands are preferred wherever possible since the payload is
	 * written with the command thus avoiding all memory registration. In
	 * addition, this allows for success events to be surpressed if
	 * FI_COMPLETION is not requested.
	 */
	ofi_genlock_lock(&txc->ep_obj->lock);
	if (idc)
		ret = cxip_rma_emit_idc(txc, buf, len, &caddr, &dfa, &idx_ext,
					vni, addr, key, data, flags, context,
					unr, tclass, tc_type);
	else
		ret = cxip_rma_emit_dma(
			txc, buf, len, desc, &caddr, &dfa, &idx_ext, vni, addr,
			key, data, flags, context, write, unr, tclass, tc_type,
			triggered, trig_thresh, trig_cntr, comp_cntr);
	ofi_genlock_unlock(&txc->ep_obj->lock);

	if (ret)
		TXC_WARN(txc,
			 "%s %s RMA %s failed: buf=%p len=%lu rkey=%#lx "
			 "roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			 unr ? "Ordered" : "Un-ordered", idc ? "IDC" : "DMA",
			 write ? "write" : "read", buf, len, key, addr,
			 caddr.nic, caddr.pid, pid_idx);
	else
		TXC_DBG(txc,
			"%s %s RMA %s emitted: buf=%p len=%lu rkey=%#lx "
			"roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			unr ? "Ordered" : "Un-ordered", idc ? "IDC" : "DMA",
			write ? "write" : "read", buf, len, key, addr,
			caddr.nic, caddr.pid, pid_idx);

	return ret;
}

/*
 * Libfabric APIs
 */
static ssize_t cxip_rma_write(struct fid_ep *fid_ep, const void *buf,
			      size_t len, void *desc, fi_addr_t dest_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len, desc,
			       dest_addr, addr, key, 0, ep->tx_attr.op_flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order,
			       context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_writev(struct fid_ep *fid_ep, const struct iovec *iov,
			       void **desc, size_t count, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key, void *context)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	size_t len;
	const void *buf;
	void *mr_desc;

	if (count == 0) {
		len = 0;
		buf = NULL;
		mr_desc = NULL;
	} else if (iov && count == 1) {
		len = iov[0].iov_len;
		buf = iov[0].iov_base;
		mr_desc = desc ? desc[0] : NULL;
	} else {
		TXC_WARN(ep->ep_obj->txc, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len, mr_desc,
			       dest_addr, addr, key, 0, ep->tx_attr.op_flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order,
			       context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_writemsg(struct fid_ep *fid_ep,
				 const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	struct cxip_txc *txc = ep->ep_obj->txc;
	size_t len;
	const void *buf;
	void *mr_desc;

	if (!msg) {
		TXC_WARN(txc, "NULL msg not supported\n");
		return -FI_EINVAL;
	}

	if (msg->rma_iov_count != 1) {
		TXC_WARN(txc, "Invalid RMA iov\n");
		return -FI_EINVAL;
	}

	if (msg->iov_count == 0) {
		len = 0;
		buf = NULL;
		mr_desc = NULL;
	} else if (msg->msg_iov && msg->iov_count == 1) {
		len = msg->msg_iov[0].iov_len;
		buf = msg->msg_iov[0].iov_base;
		mr_desc = msg->desc ? msg->desc[0] : NULL;
	} else {
		TXC_WARN(ep->ep_obj->txc, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	if (flags &
	    ~(CXIP_WRITEMSG_ALLOWED_FLAGS | FI_CXI_HRP | FI_CXI_WEAK_FENCE))
		return -FI_EBADFLAGS;

	if (flags & FI_FENCE && !(txc->attr.caps & FI_FENCE))
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_WRITE, txc, buf, len, mr_desc, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, ep->tx_attr.tclass,
			       ep->tx_attr.msg_order, msg->context, false, 0,
			       NULL, NULL);
}

ssize_t cxip_rma_inject(struct fid_ep *fid_ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len, NULL,
			       dest_addr, addr, key, 0, FI_INJECT,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order, NULL,
			       false, 0, NULL, NULL);
}

/*
 * cxip_rma_writedata() - RMA write with immediate data.
 *
 * Implements fi_writedata() using a cq_data bit approach:
 * - Initiator sets CXIP_MR_KEY_CQ_DATA_BIT in match_bits to signal writedata
 * - Initiator passes immediate data via header_data in DMA command
 * - Target MR LE ignores the cq_data bit for matching
 * - Target checks cq_data bit on PUT events to decide whether to generate
 *   a CQ completion with FI_REMOTE_CQ_DATA
 *
 * This works with both restricted and unrestricted MR modes, avoiding
 * the ordering penalties of forcing unrestricted mode for large transfers.
 * Only MRs created with FI_RMA_EVENT will generate remote CQ completions.
 */
static ssize_t cxip_rma_writedata(struct fid_ep *fid_ep, const void *buf,
				  size_t len, void *desc, uint64_t data,
				  fi_addr_t dest_addr, uint64_t addr,
				  uint64_t key, void *context)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	uint64_t flags = ep->tx_attr.op_flags | FI_REMOTE_CQ_DATA;

	if (cxip_env.disable_writedata)
		return -FI_EOPNOTSUPP;

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len, desc,
			       dest_addr, addr, key, data, flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order,
			       context, false, 0, NULL, NULL);
}

/*
 * cxip_rma_inject_writedata() - Inject RMA write with immediate data.
 *
 * Same as cxip_rma_writedata() but with FI_INJECT semantics (buffer can
 * be reused immediately after call returns).
 */
static ssize_t cxip_rma_inject_writedata(struct fid_ep *fid_ep, const void *buf,
					 size_t len, uint64_t data,
					 fi_addr_t dest_addr, uint64_t addr,
					 uint64_t key)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	uint64_t flags = FI_INJECT | FI_REMOTE_CQ_DATA;

	if (cxip_env.disable_writedata)
		return -FI_EOPNOTSUPP;

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len, NULL,
			       dest_addr, addr, key, data, flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order, NULL,
			       false, 0, NULL, NULL);
}

static ssize_t cxip_rma_read(struct fid_ep *fid_ep, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, uint64_t addr,
			     uint64_t key, void *context)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);

	return cxip_rma_common(FI_OP_READ, ep->ep_obj->txc, buf, len, desc,
			       src_addr, addr, key, 0, ep->tx_attr.op_flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order,
			       context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_readv(struct fid_ep *fid_ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	size_t len;
	const void *buf;
	void *mr_desc;

	if (count == 0) {
		len = 0;
		buf = NULL;
		mr_desc = NULL;
	} else if (iov && count == 1) {
		len = iov[0].iov_len;
		buf = iov[0].iov_base;
		mr_desc = desc ? desc[0] : NULL;
	} else {
		TXC_WARN(ep->ep_obj->txc, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	return cxip_rma_common(FI_OP_READ, ep->ep_obj->txc, buf, len, mr_desc,
			       src_addr, addr, key, 0, ep->tx_attr.op_flags,
			       ep->tx_attr.tclass, ep->tx_attr.msg_order,
			       context, false, 0, NULL, NULL);
}

static ssize_t cxip_rma_readmsg(struct fid_ep *fid_ep,
				const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_ep *ep = container_of(fid_ep, struct cxip_ep, ep);
	struct cxip_txc *txc = ep->ep_obj->txc;
	size_t len;
	const void *buf;
	void *mr_desc;

	if (!msg) {
		TXC_WARN(txc, "NULL msg not supported\n");
		return -FI_EINVAL;
	}

	if (msg->rma_iov_count != 1) {
		TXC_WARN(txc, "Invalid RMA iov\n");
		return -FI_EINVAL;
	}

	if (msg->iov_count == 0) {
		len = 0;
		buf = NULL;
		mr_desc = NULL;
	} else if (msg->msg_iov && msg->iov_count == 1) {
		len = msg->msg_iov[0].iov_len;
		buf = msg->msg_iov[0].iov_base;
		mr_desc = msg->desc ? msg->desc[0] : NULL;
	} else {
		TXC_WARN(ep->ep_obj->txc, "Invalid IOV\n");
		return -FI_EINVAL;
	}

	if (flags & ~CXIP_READMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (flags & FI_FENCE && !(txc->attr.caps & FI_FENCE))
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_READ, txc, buf, len, mr_desc, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, ep->tx_attr.tclass,
			       ep->tx_attr.msg_order, msg->context, false, 0,
			       NULL, NULL);
}

struct fi_ops_rma cxip_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = cxip_rma_read,
	.readv = cxip_rma_readv,
	.readmsg = cxip_rma_readmsg,
	.write = cxip_rma_write,
	.writev = cxip_rma_writev,
	.writemsg = cxip_rma_writemsg,
	.inject = cxip_rma_inject,
	.injectdata = cxip_rma_inject_writedata,
	.writedata = cxip_rma_writedata,
};

struct fi_ops_rma cxip_ep_rma_no_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = fi_no_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.injectdata = fi_no_rma_injectdata,
	.writedata = fi_no_rma_writedata,
};
