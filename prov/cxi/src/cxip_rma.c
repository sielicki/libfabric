/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018,2021-2023 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

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
	return cxip_cq_req_error(req, 0, ret_err,
				 cxi_event_rc(event), NULL, 0,
				 FI_ADDR_UNSPEC);
}

/*
 * cxip_rma_write_selective_completion_req() - Return request state associated
 * with all RMA write with selective completion transactions on the transmit
 * context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_rma_write_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_write_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t)txc->context;
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
static struct cxip_req *cxip_rma_read_selective_completion_req(struct cxip_txc *txc)
{
	if (!txc->rma_read_selective_completion_req) {
		struct cxip_req *req;

		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_selective_completion_cb;
		req->context = (uint64_t)txc->context;
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
		TXC_WARN(txc, CXIP_UNEXPECTED_EVENT,
			 cxi_event_to_str(event),
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
		ret = cxip_cq_req_error(req, 0, ret_err, event_rc,
					NULL, 0, FI_ADDR_UNSPEC);
		if (ret != FI_SUCCESS)
			TXC_WARN(txc, "Failed to report error: %d\n", ret);
	}

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

		req->context = (uint64_t)context;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | (write ? FI_WRITE : FI_READ) |
			(flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
		req->trig_cntr = trig_cntr;
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
					"Failed to allocate bounce buffer: %d:%s\n",
					ret, fi_strerror(-ret));
				goto err_free_cq_req;
			}

			ret = cxip_txc_copy_from_hmem(txc, NULL, req->rma.ibuf,
						      buf, len);
			if (ret){
				TXC_WARN(txc,
					 "cxip_txc_copy_from_hmem failed: %d:%s\n",
					 ret, fi_strerror(-ret));
				goto err_free_rma_buf;
			}

			dma_buf = (void *)req->rma.ibuf;
			dma_md = cxip_txc_ibuf_md(req->rma.ibuf);
		} else if (mr) {
			dma_buf = (void *)buf;
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

			dma_buf = (void *)buf;
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

	/* For writedata operations, set the cq_data bit in match_bits so
	 * the target MR can distinguish this from regular writes and
	 * generate a CQ completion with FI_REMOTE_CQ_DATA.
	 */
	if (flags & FI_REMOTE_CQ_DATA)
		dma_cmd.match_bits |= CXIP_MR_KEY_CQ_DATA_BIT;

	if (req) {
		dma_cmd.user_ptr = (uint64_t)req;
	} else {
		if (write)
			inject_req = cxip_rma_write_selective_completion_req(txc);
		else
			inject_req = cxip_rma_read_selective_completion_req(txc);

		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_rma_buf;
		}

		dma_cmd.user_ptr = (uint64_t)inject_req;
		dma_cmd.event_success_disable = 1;
	}

	if (!unr)
		dma_cmd.restricted = 1;

	if (write) {
		dma_cmd.command.opcode = C_CMD_PUT;

		/* Note: For writedata operations (FI_REMOTE_CQ_DATA), we use
		 * the emulated in-out-in protocol. The data PUT is restricted
		 * (no header_data support), and a separate 0-length unrestricted
		 * notification PUT carries the header_data. See the notification
		 * PUT code below.
		 */

		/* Triggered DMA operations have their own completion counter
		 * and the one associated with the TXC cannot be used.
		 */
		cntr = triggered ? comp_cntr : txc->write_cntr;
		if (cntr) {
			dma_cmd.event_ct_ack = 1;
			dma_cmd.ct = cntr->ct->ctn;

			if (req) {
				req->rma.cntr = cntr;
				cxip_cntr_progress_inc(cntr);
			}
		}

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			dma_cmd.flush = 1;
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

	ret = cxip_txc_emit_dma(txc, vni, cxip_ofi_to_cxi_tc(tclass),
				tc_type, trig_cntr, trig_thresh,
				&dma_cmd, flags);
	if (ret) {
		TXC_WARN(txc, "Failed to emit dma command: %d:%s\n", ret,
			 fi_strerror(-ret));
		goto err_free_rma_buf;
	}

	/* For writedata operations using emulated in-out-in protocol, emit
	 * a 0-length unrestricted PUT to the notification PTE. This PUT
	 * carries the immediate data (header_data) and the data length
	 * (remote_offset) to the target's notification LE.
	 *
	 * The notification PUT is:
	 * - Unrestricted (to deliver header_data)
	 * - 0-length (no actual data, just signaling)
	 * - Targeted at CXIP_PTL_IDX_WRITEDATA_NOTIFY PTE
	 * - match_bits = key (to match the correct MR's notification LE)
	 * - remote_offset = len (to convey the actual data length)
	 * - header_data = data (the immediate data from fi_writedata)
	 */
	if ((flags & FI_REMOTE_CQ_DATA) && write) {
		struct c_full_dma_cmd notify_cmd = {};
		union c_fab_addr notify_dfa;
		uint8_t notify_idx_ext;

		/* Build DFA for notification PTE using same target address */
		cxi_build_dfa(caddr->nic, caddr->pid, txc->pid_bits,
			      CXIP_PTL_IDX_WRITEDATA_NOTIFY, &notify_dfa,
			      &notify_idx_ext);

		notify_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		notify_cmd.command.opcode = C_CMD_PUT;
		notify_cmd.index_ext = notify_idx_ext;
		notify_cmd.dfa = notify_dfa;
		notify_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);
		/* VNI is passed as parameter to cxip_txc_emit_dma */

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

		/* Suppress ALL events for notification PUT.
		 * - event_send_disable: No send event
		 * - event_success_disable: No success event
		 * - event_ct_ack = 0: No counter event
		 * - No user_ptr needed since no events expected
		 *
		 * If the notification PUT fails on the target, the target's
		 * notification LE will generate an error event there. We
		 * cannot easily propagate that error back to the initiator
		 * since the data PUT has already completed successfully.
		 */
		notify_cmd.event_send_disable = 1;
		notify_cmd.event_success_disable = 1;
		notify_cmd.user_ptr = 0;

		ret = cxip_txc_emit_dma(txc, vni, cxip_ofi_to_cxi_tc(tclass),
					CXI_TC_TYPE_DEFAULT, NULL, 0,
					&notify_cmd, flags);
		if (ret) {
			TXC_WARN(txc, "Failed to emit notify PUT: %d:%s\n",
				 ret, fi_strerror(-ret));
			/* Data PUT already sent - can't easily rollback.
			 * The target will get the data but not the completion
			 * notification. Return the error so the initiator
			 * knows something went wrong, even though data arrived.
			 */
			return ret;
		}
	}

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

	/* IDCs must be traffic if the user requests a completion event. */
	if (flags & FI_COMPLETION) {
		req = cxip_evtq_req_alloc(&txc->tx_evtq, 0, txc);
		if (!req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc, "Failed to allocate request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err;
		}

		req->context = (uint64_t)context;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | FI_WRITE | (flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
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
			TXC_WARN(txc,
				 "cxip_txc_copy_from_hmem failed: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		idc_buf = hmem_buf;
	} else {
		idc_buf = (void *)buf;
	}

	cstate_cmd.event_send_disable = 1;
	cstate_cmd.index_ext = *idx_ext;
	cstate_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);

	if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
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
		cstate_cmd.user_ptr = (uint64_t)req;
	} else {
		inject_req = cxip_rma_write_selective_completion_req(txc);
		if (!inject_req) {
			ret = -FI_EAGAIN;
			TXC_WARN(txc,
				 "Failed to allocate inject request: %d:%s\n",
				 ret, fi_strerror(-ret));
			goto err_free_hmem_buf;
		}

		cstate_cmd.user_ptr = (uint64_t)inject_req;
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

	/* For writedata operations using emulated in-out-in protocol, emit
	 * a 0-length unrestricted DMA PUT to the notification PTE. IDC commands
	 * don't support header_data, so we must use a DMA command for the
	 * notification PUT even when the data transfer uses IDC.
	 */
	if (flags & FI_REMOTE_CQ_DATA) {
		struct c_full_dma_cmd notify_cmd = {};
		union c_fab_addr notify_dfa;
		uint8_t notify_idx_ext;

		/* Build DFA for notification PTE using same target address */
		cxi_build_dfa(caddr->nic, caddr->pid, txc->pid_bits,
			      CXIP_PTL_IDX_WRITEDATA_NOTIFY, &notify_dfa,
			      &notify_idx_ext);

		notify_cmd.command.cmd_type = C_CMD_TYPE_DMA;
		notify_cmd.command.opcode = C_CMD_PUT;
		notify_cmd.index_ext = notify_idx_ext;
		notify_cmd.dfa = notify_dfa;
		notify_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);
		/* VNI is passed as parameter to cxip_txc_emit_dma */

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

		/* Suppress ALL events for notification PUT.
		 * See comment in cxip_rma_emit_dma for rationale.
		 */
		notify_cmd.event_send_disable = 1;
		notify_cmd.event_success_disable = 1;
		notify_cmd.user_ptr = 0;

		ret = cxip_txc_emit_dma(txc, vni, cxip_ofi_to_cxi_tc(tclass),
					CXI_TC_TYPE_DEFAULT, NULL, 0,
					&notify_cmd, flags);
		if (ret) {
			TXC_WARN(txc, "Failed to emit notify PUT for IDC: %d:%s\n",
				 ret, fi_strerror(-ret));
			/* IDC PUT already sent - can't rollback.
			 * Target gets data but not the completion notification.
			 * Return error so initiator knows something went wrong.
			 */
			return ret;
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
				     uint64_t flags)
{
	/* Unoptimized keys are implemented with match bits and must always be
	 * unrestricted.
	 */
	if (!cxip_generic_is_mr_key_opt(key))
		return true;

	/* For writedata operations using emulated in-out-in protocol, the
	 * data PUT can be restricted. The notification PUT (0-length) will
	 * be unrestricted and carries the header_data. This avoids the
	 * ordering penalties of unrestricted mode for the bulk data transfer.
	 */
	if (flags & FI_REMOTE_CQ_DATA)
		return false;

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
			    bool write, bool triggered, bool unr, uint64_t flags)
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
			uint64_t msg_order, void *context,
			bool triggered, uint64_t trig_thresh,
			struct cxip_cntr *trig_cntr,
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
	 * has success events enabled (indicated by the 'events' bit in the key).
	 * Client keys don't encode this information, so we can't validate them.
	 */
	if (flags & FI_REMOTE_CQ_DATA) {
		struct cxip_mr_key mr_key = { .raw = key };

		if (!mr_key.is_prov) {
			TXC_WARN(txc, "FI_REMOTE_CQ_DATA requires provider key (FI_MR_PROV_KEY): key=0x%lx\n", key);
			return -FI_EINVAL;
		}
		if (!cxip_generic_is_mr_key_events(txc->ep_obj->caps, key)) {
			TXC_WARN(txc, "FI_REMOTE_CQ_DATA requires target MR with FI_RMA_EVENT: key=0x%lx\n", key);
			return -FI_EINVAL;
		}
	}

	unr = cxip_rma_is_unrestricted(txc, key, msg_order, write, flags);
	idc = cxip_rma_is_idc(txc, key, len, write, triggered, unr, flags);

	/* Build target network address. */
	ret = cxip_av_lookup_addr(txc->ep_obj->av, tgt_addr, &caddr);
	if (ret) {
		TXC_WARN(txc, "Failed to look up FI addr: %d:%s\n",
			 ret, fi_strerror(-ret));
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
		ret = cxip_rma_emit_dma(txc, buf, len, desc, &caddr, &dfa,
					&idx_ext, vni, addr, key, data, flags,
					context, write, unr, tclass, tc_type,
					triggered, trig_thresh,
					trig_cntr, comp_cntr);
	ofi_genlock_unlock(&txc->ep_obj->lock);

	if (ret)
		TXC_WARN(txc,
			 "%s %s RMA %s failed: buf=%p len=%lu rkey=%#lx roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			 unr ? "Ordered" : "Un-ordered",
			 idc ? "IDC" : "DMA", write ? "write" : "read",
			 buf, len, key, addr, caddr.nic, caddr.pid, pid_idx);
	else
		TXC_DBG(txc,
			"%s %s RMA %s emitted: buf=%p len=%lu rkey=%#lx roffset=%#lx nic=%#x pid=%u pid_idx=%u\n",
			unr ? "Ordered" : "Un-ordered",
			idc ? "IDC" : "DMA", write ? "write" : "read",
			buf, len, key, addr, caddr.nic, caddr.pid, pid_idx);

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

	return cxip_rma_common(FI_OP_WRITE, ep->ep_obj->txc, buf, len,
			       mr_desc, dest_addr, addr, key, 0,
			       ep->tx_attr.op_flags, ep->tx_attr.tclass,
			       ep->tx_attr.msg_order, context, false, 0, NULL,
			       NULL);
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

	if (flags & ~(CXIP_WRITEMSG_ALLOWED_FLAGS | FI_CXI_HRP |
		      FI_CXI_WEAK_FENCE))
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
