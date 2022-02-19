#include "connection.h"
#include "utils.h"

#include <random>
#include <iostream>


#include <errno.h>

#define _DCB_COUT_(x) // std::cout << x

namespace bbts {
namespace ib {

struct bbts_dest_t {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
};

// TODO: it'd really be better to have a set of qps dedicated to just rdma writes.
// Now rdma writes are shared with the qps that also send messages
struct bbts_context_t {
  ibv_context  *context;
  ibv_pd    *pd;
  ibv_cq    *cq;
  std::vector<ibv_qp_ptr> qps;
  ibv_srq   *srq;
  ibv_port_attr portinfo;

  bbts_message_t *recv_msgs;
  ibv_mr *recv_msgs_mr;

  bbts_message_t *send_msgs;
  ibv_mr *send_msgs_mr;

  uint8_t num_rank;
  uint32_t num_send_per_qp; // the number of send messages that can be added to each
                            // queue pair
  uint32_t num_write_per_qp; // the number of rmda writes that can be added to each
                             // queue pair

  uint32_t get_num_recv() const {
    // If every other node sends as many messages as they can here,
    // that is how many num_recv there must be
    return get_num_qp()*num_send_per_qp;
  }
  uint32_t get_num_total_send() const {
    // This is the same as num_recv
    return get_num_qp()*num_send_per_qp;
  }
  uint8_t get_num_qp() const {
    // there is one qp for every rank except this rank
    return num_rank - 1;
  }
  uint32_t get_num_total_write() const {
    return get_num_qp()*num_write_per_qp;
  }
};

void bbts_post_rdma_write(
  ibv_qp *qp,
  uint64_t wr_id,
  void* local_addr, uint32_t local_size, uint32_t local_key,
  uint64_t remote_addr, uint32_t remote_key)
{
  ibv_sge list = {
    .addr  = (uint64_t)local_addr,
    .length = local_size,
    .lkey  = local_key
  };
  ibv_send_wr wr = {
    .wr_id      = wr_id,
    .sg_list    = &list,
    .num_sge    = 1,
    .opcode     = IBV_WR_RDMA_WRITE,
    .send_flags = IBV_SEND_SIGNALED,
    .wr = {
      .rdma = {
        .remote_addr = remote_addr,
        .rkey        = remote_key
      }
    }
  };

  ibv_send_wr *bad_wr;

  if(errno = ibv_post_send(qp, &wr, &bad_wr)) {
    perror("ibv_post_send for rdma write ");
    throw std::runtime_error("ibv_post_send");
  }
}

void bbts_post_rdma_write(
  ibv_qp *qp,
  bbts_rdma_write_t const& r)
{
  bbts_post_rdma_write(
    qp,
    r.wr_id,
    r.local_addr, r.local_size, r.local_key,
    r.remote_addr, r.remote_key);
}

void bbts_post_send_message(
  ibv_qp *qp,
  uint64_t wr_id,
  uint32_t local_key,
  bbts_message_t const& msg)
{
  ibv_sge list = {
    .addr  = (uint64_t)(&msg),
    .length = sizeof(bbts_message_t),
    .lkey = local_key
  };
  ibv_send_wr wr = {
    .wr_id      = wr_id,
    .sg_list    = &list,
    .num_sge    = 1,
    .opcode     = IBV_WR_SEND,
    .send_flags = IBV_SEND_SIGNALED
  };
  // INLINE:   copy the buffer, making the buffer immediately available
  // SIGNALED: make sure a work completion gets added to the completion queue

  ibv_send_wr *bad_wr;

  if(errno = ibv_post_send(qp, &wr, &bad_wr)) {
    perror("FAILED TO POST SEND MESSAGE ");
    throw std::runtime_error("ibv_post_send");
  }
}

void bbts_post_recv_message(ibv_srq *srq, uint32_t local_key, bbts_message_t& msg)
{
  ibv_sge list = {
    .addr  = (uint64_t)(&msg),
    .length = sizeof(bbts_message_t),
    .lkey  = local_key
  };
  ibv_recv_wr wr = {
    .wr_id      = 0,
    .sg_list    = &list,
    .num_sge    = 1,
  };
  ibv_recv_wr *bad_wr;

  if(ibv_post_srq_recv(srq, &wr, &bad_wr)) {
    throw std::runtime_error("Error in post recv");
  }
}

bbts_context_t* bbts_init_context(
    std::string dev_name,
    int32_t rank,
    unsigned int num_rank,
    uint8_t ib_port = 1,
    uint32_t num_send_per_qp = 0,
    uint32_t num_write_per_qp = 0)
{
  // Give default values for when num_send_per_qp is not specified
  // or given an invalid zero value
  if(num_send_per_qp == 0) {
    // Max ability to send 128 messages to a node
    num_send_per_qp = 128;
  }
  if(num_write_per_qp == 0) {
    // Max ability to do this many rdma writes
    num_write_per_qp = 32;
  }

  bbts_context_t *ctx = new bbts_context_t;
  ctx->num_rank         = num_rank;
  ctx->num_send_per_qp  = num_send_per_qp;
  ctx->num_write_per_qp = num_write_per_qp;
  ctx->qps              = std::vector<ibv_qp_ptr>(num_rank);

  uint32_t num_recv        = ctx->get_num_recv();
  uint32_t num_total_send  = ctx->get_num_total_send();
  uint32_t num_total_write = ctx->get_num_total_write();

  ibv_device  *ib_dev = NULL;
  ibv_device **dev_list = ibv_get_device_list(NULL);
  for(int i = 0; dev_list[i]; ++i) {
    const char* name = ibv_get_device_name(dev_list[i]);
    if(dev_name == name) {
      ib_dev = dev_list[i];
    }
  }
  if(!ib_dev) {
    goto clean_dev_list;
  }

  ctx->context = ibv_open_device(ib_dev);
  if(!ctx->context) {
    goto clean_ctx;
  }

  ctx->pd = ibv_alloc_pd(ctx->context);
  if (!ctx->pd) {
    goto clean_device;
  }

  ctx->recv_msgs = new bbts_message_t[num_recv];
  ctx->recv_msgs_mr = ibv_reg_mr(
      ctx->pd,
      ctx->recv_msgs, num_recv*sizeof(bbts_message_t),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if(!ctx->recv_msgs_mr) {
    goto clean_pd;
  }

  ctx->send_msgs = new bbts_message_t[num_total_send];
  ctx->send_msgs_mr = ibv_reg_mr(
      ctx->pd,
      ctx->send_msgs, num_total_send*sizeof(bbts_message_t),
      IBV_ACCESS_LOCAL_WRITE);
  if(!ctx->send_msgs_mr) {
    goto clean_recv_mr;
  }

  ctx->cq = ibv_create_cq(
    ctx->context,
    num_total_write + num_total_send + num_recv,
    NULL, NULL, 0);
  if (!ctx->cq) {
    goto clean_send_mr;
  }

  {
		ibv_srq_init_attr attr = {
			.attr = {
				.max_wr  = num_recv,
				.max_sge = 1
			}
		};
    ctx->srq = ibv_create_srq(ctx->pd, &attr);
    if(!ctx->srq) {
      goto clean_cq;
    }
  }

  {
    ibv_qp_init_attr init_attr = {
      .send_cq = ctx->cq,
      .recv_cq = ctx->cq,
      .srq     = ctx->srq,
      .cap     = {
        .max_send_wr  = num_send_per_qp + num_write_per_qp,
        .max_recv_wr  = 0,
        .max_send_sge = 1,
        .max_recv_sge = 1
      },
      .qp_type = IBV_QPT_RC
    };

    int i;
    for(i = 0; i != num_rank; ++i) {
      if(i == rank) {
        continue;
      }
      ctx->qps[i] = ibv_create_qp(ctx->pd, &init_attr);
      if(!ctx->qps[i]) {
        i -= 1;
        for(; i >= 0; --i) {
          if(i == rank)
            continue;
          ibv_destroy_qp(ctx->qps[i]);
        }
        goto clean_srq;
      }
    }
  }

  {
    ibv_qp_attr attr = {
      .qp_state        = IBV_QPS_INIT,
      // TODO: not sure if specifying access flags is necessary
      .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
      .pkey_index      = 0,
      .port_num        = ib_port,
    };

    for(int i = 0; i != num_rank; ++i) {
      if(i == rank) {
        continue;
      }
      if (ibv_modify_qp(ctx->qps[i], &attr,
            IBV_QP_STATE              |
            IBV_QP_PKEY_INDEX         |
            IBV_QP_PORT               |
            IBV_QP_ACCESS_FLAGS)) {
        goto clean_qps;
      }
    }
  }

  ibv_free_device_list(dev_list);
  return ctx;

clean_qps:
  for(int i = 0; i != num_rank; ++i) {
    if(i == rank)
      continue;
    ibv_destroy_qp(ctx->qps[i]);
  }

clean_srq:
  ibv_destroy_srq(ctx->srq);

clean_cq:
  ibv_destroy_cq(ctx->cq);

clean_send_mr:
  ibv_dereg_mr(ctx->send_msgs_mr);
  delete[] ctx->send_msgs;

clean_recv_mr:
  ibv_dereg_mr(ctx->recv_msgs_mr);
  delete[] ctx->recv_msgs;

clean_pd:
  ibv_dealloc_pd(ctx->pd);

clean_device:
  ibv_close_device(ctx->context);

clean_ctx:
  free(ctx);

clean_dev_list:
  ibv_free_device_list(dev_list);

  return nullptr;
}

// Move the queue pair through the states to get it into the
// Ready to Send state.
bool bbts_connect_context_helper(
  ibv_qp *qp,
  uint8_t ib_port,
  int my_psn,
	bbts_dest_t const& dest)
{
	ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= IBV_MTU_1024,
		.rq_psn			= dest.psn,
		.dest_qp_num		= dest.qpn,
		.ah_attr		= {
			.dlid		= dest.lid,
			.sl		= 0,
			.src_path_bits	= 0,
			.is_global	= 0,
			.port_num	= ib_port
		},
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 12
	};

	if (ibv_modify_qp(qp, &attr,
			  IBV_QP_STATE              |
			  IBV_QP_AV                 |
			  IBV_QP_PATH_MTU           |
			  IBV_QP_DEST_QPN           |
			  IBV_QP_RQ_PSN             |
			  IBV_QP_MAX_DEST_RD_ATOMIC |
			  IBV_QP_MIN_RNR_TIMER)) {
		return false;
	}

	attr.qp_state	    = IBV_QPS_RTS;
	attr.timeout	    = 14;
	attr.retry_cnt	    = 7;
	attr.rnr_retry	    = 7;
	attr.sq_psn	    = my_psn;
	attr.max_rd_atomic  = 1;
	if (ibv_modify_qp(qp, &attr,
			  IBV_QP_STATE              |
			  IBV_QP_TIMEOUT            |
			  IBV_QP_RETRY_CNT          |
			  IBV_QP_RNR_RETRY          |
			  IBV_QP_SQ_PSN             |
			  IBV_QP_MAX_QP_RD_ATOMIC)) {
		return false;
	}

	return true;
}

uint32_t random_psn() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<uint32_t> d;
  return d(gen);
}

bool bbts_client_exch_dest_and_connect(
  bbts_context_t *ctx,
  int qp_index,
  std::string servername,
  uint8_t ib_port, int port,
  bbts_dest_t& rem_dest)
{
  bbts_dest_t my_dest = {
    .lid = ctx->portinfo.lid,
    .qpn = ctx->qps[qp_index]->qp_num,
    .psn = random_psn()
  };
  bool success = client_exch(servername, port, my_dest, rem_dest);
  if(!success) {
    return false;
  }

  if(!bbts_connect_context_helper(ctx->qps[qp_index], ib_port, my_dest.psn, rem_dest)) {
    return false;
  }

  return true;
}

bool bbts_server_exch_dest_and_connect(
  bbts_context_t *ctx,
  int qp_index,
  uint8_t ib_port, int port,
  bbts_dest_t& rem_dest)
{
  bbts_dest_t my_dest = {
    .lid = ctx->portinfo.lid,
    .qpn = ctx->qps[qp_index]->qp_num,
    .psn = random_psn()
  };
  bool success = server_exch(
        port,
        [&]()
        {
          return bbts_connect_context_helper(
                  ctx->qps[qp_index], ib_port, my_dest.psn, rem_dest);
        },
        my_dest, rem_dest);
  if(!success) {
    return false;
  }

  return true;
}

int bbts_connect_context(
    bbts_context_t *ctx,
    int32_t rank,
    std::vector<std::string> ips,
    uint8_t ib_port = 1, int port = 18515)
{
  std::vector<bbts_dest_t> rem_dests(ctx->num_rank);

  if (ibv_query_port(ctx->context, ib_port, &ctx->portinfo)) {
    return 1;
  }

  // Both the server and the client exch functions call bbts_connect_context_helper,
  // but they do so at different points.
  // This code was built off of an experiment which was built off of with libibverbs
  // pingpong example, and that is what they did.
  // It is unclear to me why that is the case.
  //
  for(int i = 0; i != ctx->num_rank; ++i) {
    if(i == rank)
      continue;

    bool success_exch =
      (i >= rank)
      ? bbts_server_exch_dest_and_connect(ctx, i,         ib_port, port, rem_dests[i])
      : bbts_client_exch_dest_and_connect(ctx, i, ips[i], ib_port, port, rem_dests[i]);
    if(!success_exch){
      return 1;
    }
  }

  // PRINT DEST INFO HERE
  for(int i = 0; i != ctx->num_rank; ++i) {
    if(i == rank) {
      printf("local LID: 0x%04x\n",
             ctx->portinfo.lid);
    } else {
      printf("remote address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
             rem_dests[i].lid, rem_dests[i].qpn, rem_dests[i].psn);
    }
  }

  return 0;
}


connection_t::connection_t(
  std::string dev_name,
  int32_t rank,
  std::vector<std::string> ips):
    rank(rank),
    current_recv_msg(0),
    send_wr_cnts(ips.size()),
    pending_msgs(ips.size()),
    write_wr_cnts(ips.size()),
    pending_writes(ips.size())
{
  int num_rank = ips.size();

  bbts_context_t *context = bbts_init_context(
    dev_name,
    rank,
    num_rank);

  if(!context) {
    throw std::runtime_error("make_connection: couldn't init context");
  }

  // add the available send messages
  for(int i = 0; i != context->get_num_total_send(); ++i) {
    available_send_msgs.push(i);
  }

  // we can send this many items
  std::fill(send_wr_cnts.begin(), send_wr_cnts.end(), context->num_send_per_qp);
  // we can do this many rdma writes
  std::fill(write_wr_cnts.begin(), write_wr_cnts.end(), context->num_write_per_qp);

  // post a receive before setting up a connection so that
  // when the connection is started, the receiving can occur
  for(int which_recv = 0; which_recv != context->get_num_recv(); ++which_recv) {
    bbts_post_recv_message(
      context->srq,
      context->recv_msgs_mr->lkey,
      context->recv_msgs[which_recv]);
  }

  if(bbts_connect_context(context, rank, ips)) {
    throw std::runtime_error("make_connection: couldn't connect context");
  }

  destruct = false;

  this->num_rank         = context->num_rank;
  this->num_recv         = context->get_num_recv();
  this->num_send_per_qp  = context->num_send_per_qp;
  this->num_write_per_qp = context->num_write_per_qp;

  this->recv_msgs      = context->recv_msgs;
  this->recv_msgs_mr   = context->recv_msgs_mr;
  this->send_msgs      = context->send_msgs;
  this->send_msgs_mr   = context->send_msgs_mr;

  this->context               = context->context;
  this->completion_queue      = context->cq;
  this->protection_domain     = context->pd;
  this->queue_pairs           = context->qps;
  this->shared_recv_queue     = context->srq;

  delete context;

  poll_thread = std::thread(&connection_t::poll, this);
}

connection_t::~connection_t() {
  destruct = true;
  poll_thread.join();

  for(int i = 0; i != rank; ++i) {
    if(i == rank)
      continue;
    ibv_destroy_qp(this->queue_pairs[i]);
  }
  ibv_destroy_srq(this->shared_recv_queue);
  ibv_destroy_cq(this->completion_queue);
  ibv_dereg_mr(this->send_msgs_mr);
  ibv_dereg_mr(this->recv_msgs_mr);
  delete[] send_msgs;
  delete[] recv_msgs;
  ibv_dealloc_pd(this->protection_domain);
  ibv_close_device(this->context);
}

std::future<bool> connection_t::send_bytes_(
  int32_t dest_rank, tag_t send_tag, bytes_t bytes, bool imm)
{
  if(dest_rank == rank) {
    throw std::invalid_argument("destination rank is same as current rank");
  }
  if(send_tag == 0) {
    throw std::invalid_argument("tag cannot be zero");
  }
  std::lock_guard<std::mutex> lk(send_m);
  send_init_queue.push_back({
    tag_rank_t{send_tag, dest_rank},
    std::unique_ptr<send_item_t>(new send_item_t(this, bytes, imm))
  });
  return send_init_queue.back().second->get_future();
}


std::future<bool> connection_t::send_bytes(int32_t dest_rank, tag_t send_tag, bytes_t bytes){
  return send_bytes_(dest_rank, send_tag, bytes, true);
}

std::future<recv_bytes_t> connection_t::recv_bytes(tag_t recv_tag){
  if(recv_tag == 0) {
    throw std::invalid_argument("tag cannot be zero");
  }
  std::lock_guard<std::mutex> lk(recv_m);
  recv_init_queue.push_back({
    recv_tag,
    std::unique_ptr<recv_item_t>(new recv_item_t(true))
  });
  return recv_init_queue.back().second->get_future_bytes();
}

std::future<bool> connection_t::send_bytes_wait(int32_t dest_rank, tag_t send_tag, bytes_t bytes){
  return send_bytes_(dest_rank, send_tag, bytes, false);
}

std::future<bool> connection_t::recv_bytes_wait(tag_t recv_tag, bytes_t bytes) {
  if(recv_tag == 0) {
    throw std::invalid_argument("tag cannot be zero");
  }
  std::lock_guard<std::mutex> lk(recv_m);
  recv_init_queue.push_back({
    recv_tag,
    std::unique_ptr<recv_item_t>(new recv_item_t(bytes))
  });
  return recv_init_queue.back().second->get_future_complete();
}

connection_t::send_item_t::send_item_t(connection_t *connection, bytes_t b, bool imm):
  bytes(b), imm(imm)
{
  bytes_mr = ibv_reg_mr(
      connection->protection_domain,
      bytes.data, bytes.size,
      IBV_ACCESS_LOCAL_WRITE);
  if(!bytes_mr) {
    throw std::runtime_error("couldn't register mr for send bytes");
  }
}

void connection_t::send_item_t::send(
  connection_t *connection,
  tag_t tag,
  int32_t dest_rank,
  uint64_t remote_addr,
  uint32_t remote_key)
{
  _DCB_COUT_("post rdma write to dest " << dest_rank << ", tag " << tag << ", key " <<
      remote_key << ", address " << remote_addr << std::endl);

  // TODO: what happens when size bigger than 2^31...
  // TODO: what wrid?
  connection->post_rdma_write(dest_rank,
  {
    .wr_id       = tag,
    .local_addr  = bytes.data,
    .local_size  = bytes.size,
    .local_key   = bytes_mr->lkey,
    .remote_addr = remote_addr,
    .remote_key  = remote_key
  });
}

std::future<recv_bytes_t> connection_t::recv_item_t::get_future_bytes() {
  if(!own) {
    throw std::runtime_error("cannot call get_future_bytes when recv does not own bytes");
  }
  if(!valid_promise) {
    throw std::runtime_error("invalid promise in recv item");
  }
  return pr_bytes.get_future();
}

std::future<bool> connection_t::recv_item_t::get_future_complete() {
  if(own) {
    throw std::runtime_error("cannot call get_future_complete when recv owns bytes");
  }
  if(!valid_promise) {
    throw std::runtime_error("invalid promise in recv item");
  }
  return pr_complete.get_future();
}

void connection_t::recv_item_t::init(connection_t *connection, uint64_t size){
  if(own) {
    bytes.data = (void*)(new char[size]);
    bytes.size = size;
  }

  bytes_mr = ibv_reg_mr(
      connection->protection_domain,
      bytes.data, bytes.size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if(!bytes_mr) {
    perror("register mr fail recv bytes ");
    throw std::runtime_error("couldn't register mr for recv bytes");
  }
}

void connection_t::post_send(int32_t dest_rank, bbts_message_t const& msg) {
  if(send_wr_cnts[dest_rank] == 0) {
    pending_msgs[dest_rank].push(msg);
  } else {
    send_wr_cnts[dest_rank] -= 1;

    int i = available_send_msgs.front();
    available_send_msgs.pop();

    send_msgs[i] = msg;
    bbts_post_send_message(queue_pairs[dest_rank], i, send_msgs_mr->lkey, send_msgs[i]);
  }
}

void connection_t::post_rdma_write(int32_t dest_rank, bbts_rdma_write_t const& r) {
  if(write_wr_cnts[dest_rank] == 0) {
    _DCB_COUT_("rdma write to pending writes" << std::endl);
    pending_writes[dest_rank].push(r);
  } else {
    _DCB_COUT_("rdma write direct" << std::endl);
    write_wr_cnts[dest_rank] -= 1;

    bbts_post_rdma_write(queue_pairs[dest_rank], r);
  }
}

void connection_t::post_open_send(int32_t dest_rank, tag_t tag, uint64_t size, bool imm){
  _DCB_COUT_("post open send to dest " << dest_rank << ", tag " << tag << std::endl);
  this->post_send(dest_rank,
  {
    .type = bbts_message_t::message_type::open_send,
    .rank      = dest_rank,
    .from_rank = rank,
    .tag = tag,
    .m = {
      .open_send {
        .immediate = imm,
        .size = size,
      }
    }
  });
}

void connection_t::post_open_recv(
  int32_t dest_rank, tag_t tag, uint64_t addr, uint64_t size, uint32_t key)
{
  _DCB_COUT_("post open recv to dest " << dest_rank << ", tag " << tag << ", key " << key << ", address " << addr << std::endl);
  this->post_send(dest_rank,
  {
    .type = bbts_message_t::message_type::open_recv,
    .rank      = dest_rank,
    .from_rank = rank,
    .tag = tag,
    .m = {
      .open_recv {
        .addr = addr,
        .size = size,
        .key = key
      }
    }
  });
}

void connection_t::post_close(int32_t dest_rank, tag_t tag){
  _DCB_COUT_("post close to dest " << dest_rank << ", tag " << tag << std::endl);
  this->post_send(dest_rank,
  {
    .type = bbts_message_t::message_type::close_send,
    .rank      = dest_rank,
    .from_rank = rank,
    .tag = tag
  });
}


void connection_t::post_fail_send(int32_t dest_rank, tag_t tag) {
  _DCB_COUT_("post fail send, to dest  " << dest_rank << ", tag " << tag << std::endl);
  this->post_send(dest_rank,
  {
    .type = bbts_message_t::message_type::fail_send,
    .rank      = dest_rank,
    .from_rank = rank,
    .tag = tag
  });
}

int connection_t::get_recv_rank(ibv_wc const& wc)
{
  for(int i = 0; i != num_rank; ++i) {
    if(i == rank)
      continue;
    if(queue_pairs[i]->qp_num == wc.qp_num) {
      return i;
    }
  }
  throw std::runtime_error("could not get rank from work completion");
}

void connection_t::handle_message(bbts_message_t const& msg) {
  int32_t const& recv_rank = msg.from_rank;
  if(msg.type == bbts_message_t::message_type::open_send) {
    _DCB_COUT_("recvd open send from " << recv_rank << ", tag " << msg.tag << std::endl);

    // There are two cases: this send needs a recv now or later.
    // If now and the the recv_item_t doesn't exist, create one.
    auto iter = recv_items.find(msg.tag);
    if(iter == recv_items.end()) {
      if(msg.m.open_send.immediate) {
        iter = recv_items.insert({
          msg.tag,
          std::unique_ptr<recv_item_t>(new recv_item_t(false))
        }).first;
      } else {
        // save the message for when recv_items is called.
        pending_recvs.insert({
          msg.tag,
          msg});
        return;
      }
    }
    // Register memory and maybe also allocate memory.
    recv_item_t& item = *(iter->second);
    item.init(this, msg.m.open_recv.size);

    // sending a message back to the location that sent this message
    this->post_open_recv(
      recv_rank, msg.tag,
      (uint64_t)item.bytes.data,
      item.bytes.size,
      item.bytes_mr->rkey);
  } else if(msg.type == bbts_message_t::message_type::open_recv) {
    _DCB_COUT_("recvd open recv from " << recv_rank << ", tag " << msg.tag << std::endl);

    // Do an rdma write if there is a send request available.
    // Otherwise, save the message for when send_bytes is called.
    auto iter = send_items.find({msg.tag, recv_rank});
    if(iter == send_items.end()) {
      pending_sends.insert({
        {msg.tag, recv_rank},
        msg
      });
    } else {
      send_item_t& item = *(iter->second);
      item.send(this, msg.tag, recv_rank, msg.m.open_recv.addr, msg.m.open_recv.key);
    }
  } else if(msg.type == bbts_message_t::message_type::close_send) {
    _DCB_COUT_("recvd close send from " << recv_rank << ", tag " << msg.tag << std::endl);

    // the recv item knows it has been written to so set the future
    // on the recv_item delete it UNLESS the promise is invalid
    auto iter = recv_items.find(msg.tag);
    if(iter == recv_items.end()) {
      throw std::runtime_error("invalid tag in close message");
    }
    recv_item_t& item = *(iter->second);
    if(item.valid_promise) {
      if(item.is_set) {
        throw std::runtime_error("invalid promise state");
      }
      if(item.own) {
        item.pr_bytes.set_value(item.bytes);
      } else {
        item.pr_complete.set_value(true);
      }
      recv_items.erase(iter);
    } else {
      item.is_set = true;
    }
  } else if(msg.type == bbts_message_t::message_type::fail_send) {
    _DCB_COUT_("recvd fail send from " << recv_rank << ", tag " << msg.tag << std::endl);

    // set the future item to false and delete it
    auto iter = send_items.find({msg.tag, recv_rank});
    if(iter == send_items.end()) {
      throw std::runtime_error("there should be a send item here");
    }
    send_item_t& item = *(iter->second);
    item.pr.set_value(false);
    send_items.erase(iter);
  } else {
    throw std::runtime_error("invalid message type");
  }
}

void connection_t::poll(){
  ibv_wc work_completion;
  while(!destruct){
    // It's not important that the thread catches the destruct exactly.
    for(int i = 0; i != 1000000; ++i) {
      // 1. empty send_init_queue
      // 2. empty recv_init_queue
      // 3. receive a work request

      // 1. empty send_init_queue
      {
        std::lock_guard<std::mutex> lk(send_m);
        for(auto && item: send_init_queue) {
          // does an rdma write already have to happen?
          auto iter = pending_sends.find(item.first);
          if(iter == pending_sends.end()) {
            // no? send an open send so you can get the remote info
            this->post_open_send(
              item.first.rank,
              item.first.tag,
              item.second->bytes.size,
              item.second->imm);
            send_items.insert(std::move(item));
          } else {
            // yes? the remote info is already available, so do a remote write
            bbts_message_t& msg = iter->second;
            send_item_t& send_item = *(item.second);
            send_item.send(this, msg.tag, msg.rank, msg.m.open_recv.addr, msg.m.open_recv.key);
            pending_sends.erase(iter);
            // still add it to send items so it can inform the peer when a
            // write has happened
            send_items.insert(std::move(item));
          }
        }
        send_init_queue.resize(0);
      }

      // 2. empty recv_init_queue
      {
        std::lock_guard<std::mutex> lk(recv_m);
        for(auto && item: recv_init_queue) {
          tag_t tag = item.first;

          // We consider the 4 cases where the
          //   corresponding open send may or may not have been provided and
          //   a corresponding recv was or was not already posted
          auto iter_recv = recv_items.find(tag);
          auto iter_pend = pending_recvs.find(tag);
          if(iter_recv != recv_items.end() && iter_pend != pending_recvs.end()) {
            // This should never occur
            throw std::runtime_error("don't mix and match send and recv with wait variant");
          } else if(iter_recv != recv_items.end()) {
            // There are currently two recv items, but since (assumption)
            // send_bytes and recv_bytes are only available once for each tag,
            // this means that the only relevant promise is the one from
            // the recv_bytes call in item--the one in recv_items is because
            // an OpenRecv command has already been sent.

            // If an OpenRecv command has been sent, then the recv_item must own the bytes
            if(!iter_recv->second->own) {
              throw std::runtime_error("shouldn't own the bytes");
            }
            // If the bytes are alredy set,
            //   (1) set the promsie and
            //   (2) remove the entry in recv_items.
            // Otherwise the bytes are not set,
            //   move the relevant promise into recv_items
            if(iter_recv->second->is_set) {
              item.second->pr_bytes.set_value(iter_recv->second->bytes);
              recv_items.erase(iter_recv);
            } else {
              iter_recv->second->pr_bytes = std::move(item.second->pr_bytes);
              iter_recv->second->valid_promise = true;
            }
          } else if(iter_pend != pending_recvs.end()) {
            // An open recv command hasn't been sent but we have the available information to
            // send it.
            // This also means that item must own it's bytes and have been created from
            // recv_bytes_wait, not recv_bytes.
            recv_item_t& r = *item.second;
            if(r.own) {
              throw std::runtime_error("this recv item was not called with wait variant");
            }

            bbts_message_t const& msg = iter_pend->second;

            // Are there enough bytes available?
            if(r.bytes.size < msg.m.open_recv.size) {
              // uh-oh. This can't be done.
              // On this side, we tell the recv item we failed
              r.pr_complete.set_value(false);
              // On that side, we tell the waiting send that this isn't gonna happen
              this->post_fail_send(msg.from_rank, tag);
            } else {
              // init the item (this won't allocate anything since it doesn't own bytes)
              r.init(this, msg.m.open_recv.size);

              // sending a message back to the location that sent this message
              this->post_open_recv(
                msg.from_rank,
                tag,
                (uint64_t)r.bytes.data,
                r.bytes.size,
                r.bytes_mr->rkey);

              // now wait for the write to be finished in here
              recv_items.insert(std::move(item));
            }

            // pending recvs has been handled
            pending_recvs.erase(iter_pend);
          } else {
            recv_items.insert(std::move(item));
          }
        }
        recv_init_queue.resize(0);
      }

      // 3. is there anything in the receive queue? If so, handle it
      //    TODO: tidy this up; create a parse work_completion function..
      int ne = ibv_poll_cq(completion_queue, 1, &work_completion);
      while(ne != 0) {
        if(ne < 0) {
          throw std::runtime_error("ibv_poll_cq error");
        }
        if(int err = work_completion.status) {
          _DCB_COUT_("work completion status " << err << std::endl);
          throw std::runtime_error("work completion error");
        }

        bool is_send = work_completion.opcode == 0;
        bool is_recv = work_completion.opcode == 128;
        bool is_rdma_write = work_completion.opcode == 1;
        auto wr_id = work_completion.wr_id;
        int32_t wc_rank = get_recv_rank(work_completion);
        if(is_rdma_write && wr_id > 0) {
          _DCB_COUT_("finished rdma write " << wr_id << std::endl);
          // an rdma write occured
          auto tag = wr_id;
          auto dest_rank = wc_rank;

          // inform the destination we are done
          post_close(dest_rank, tag);

          // update the write_wr_cnt
          write_wr_cnts[dest_rank] += 1;

          if(!pending_writes[dest_rank].empty()) {
            _DCB_COUT_("rdma write from queue" << std::endl);
            // making an rdma write, decement it back
            write_wr_cnts[dest_rank] -= 1;
            // get the meta data and do the write
            bbts_post_rdma_write(
              queue_pairs[dest_rank],
              pending_writes[dest_rank].front());
            // and remove the item from the queue
            pending_writes[dest_rank].pop();
          }
        } else if(is_send) {
          int i = wr_id;
          if(send_msgs[i].type == bbts_message_t::message_type::close_send) {
            _DCB_COUT_("finished send close tag " << send_msgs[i].tag << std::endl);
            auto iter = send_items.find({send_msgs[i].tag, wc_rank});
            if(iter == send_items.end()) {
              throw std::runtime_error("can't close send item, how can it not be here");
            }
            send_item_t& item = *(iter->second);
            item.pr.set_value(true);
            send_items.erase(iter);
          }

          int32_t dest_rank = send_msgs[i].rank;

          // add the index to the send msg now that this send_msg is available
          available_send_msgs.push(i);

          // update the send_wr_cnt
          send_wr_cnts[dest_rank] += 1;

          // check the queue
          if(!pending_msgs[dest_rank].empty()) {
            // making another send, decrement the send_wr_cnt
            send_wr_cnts[dest_rank] -= 1;

            // get an open send_msg object
            // (which is why it was important to add to available_send_msgs
            //  above before getting here)
            int which = available_send_msgs.front();
            available_send_msgs.pop();

            // write the message to be sent to the avilable send_msgs memory
            send_msgs[which] = pending_msgs[dest_rank].front();
            pending_msgs[dest_rank].pop();

            bbts_post_send_message(
              queue_pairs[dest_rank],
              which,
              send_msgs_mr->lkey,
              send_msgs[which]);
          }
        } else if(is_recv) {
          // a message has been recvd
          // so post a recv back and handle the message

          // copy the message here so msg_recv can be used again
          // in the post recv
          bbts_message_t msg = recv_msgs[current_recv_msg];

          bbts_post_recv_message(
            shared_recv_queue, recv_msgs_mr->lkey, recv_msgs[current_recv_msg]);

          current_recv_msg++;
          if(current_recv_msg == num_recv) {
            current_recv_msg = 0;
          }

          this->handle_message(msg);
        } else {
          throw std::runtime_error("unhandled item from recv queue");
        }
        ne = ibv_poll_cq(completion_queue, 1, &work_completion);
      }
    }
  }
}

} // namespace ib
} // namespace bbts
