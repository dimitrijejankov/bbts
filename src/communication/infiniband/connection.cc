#include "connection.h"
#include "utils.h"

#include <random>
#include <iostream>

namespace bbts {
namespace ib {

struct bbts_dest_t {
  uint16_t lid;
  uint32_t qpn;
  uint32_t psn;
};

struct bbts_context_t {
  ibv_context  *context;
  ibv_pd    *pd;
  ibv_cq    *cq;
  std::vector<ibv_qp_ptr> qps;
  ibv_srq   *srq;
  ibv_port_attr portinfo;
  // the first num_qp are recv messages, then the nex num_qp are send messages
  bbts_message_t *msgs;
  ibv_mr *msgs_mr;
  std::vector<uint64_t> msg_remote_addr;
  std::vector<uint32_t> msg_remote_key;
  uint8_t num_qp;
};

struct bbts_exch_t {
  bbts_dest_t dest;
  uint64_t addr;
  uint32_t rem_key;
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

  if(ibv_post_send(qp, &wr, &bad_wr)) {
    throw std::runtime_error("ibv_post_send");
  }
}

void bbts_post_send(
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
    .opcode     = IBV_WR_SEND,
    .send_flags = IBV_SEND_SIGNALED,
    .wr = {
      .rdma = {
        .remote_addr = remote_addr,
        .rkey        = remote_key
      }
    }
  };

  ibv_send_wr *bad_wr;

  if(ibv_post_send(qp, &wr, &bad_wr)) {
    throw std::runtime_error("ibv_post_send");
  }
}

void bbts_post_recv(
  ibv_srq *srq,
  uint64_t wr_id,
  void* local_addr, uint32_t local_size, uint32_t local_key,
  uint32_t expected_remote_key)
{
  // expected_remote_key is only for debugging purposes

  ibv_sge list = {
    .addr  = (uint64_t)local_addr,
    .length = local_size,
    .lkey  = local_key
  };
  ibv_recv_wr wr = {
    .wr_id      = wr_id,
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
    unsigned int num_qp,
    uint8_t ib_port = 1)
{
  bbts_context_t *ctx = new bbts_context_t;
  ctx->num_qp = num_qp;
  ctx->qps             = std::vector<ibv_qp_ptr>(num_qp);
  ctx->msg_remote_addr = std::vector<uint64_t>(num_qp);
  ctx->msg_remote_key  = std::vector<uint32_t>(num_qp);

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

  ctx->msgs = new bbts_message_t[2*num_qp];
  ctx->msgs_mr = ibv_reg_mr(
      ctx->pd,
      ctx->msgs, 2*num_qp*sizeof(bbts_message_t),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if(!ctx->msgs_mr) {
    goto clean_pd;
  }

  ctx->cq = ibv_create_cq(ctx->context, 1000, NULL, NULL, 0);
  //ctx->cq = ibv_create_cq(ctx->context, 2*num_qp, NULL, NULL, 0);
  if (!ctx->cq) {
    goto clean_mr;
  }

  {
		ibv_srq_init_attr attr = {
			.attr = {
				.max_wr  = 1000, //2*num_qp,
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
        .max_send_wr  = 1000,
        .max_recv_wr  = 1000,
        .max_send_sge = 1,
        .max_recv_sge = 1
      },
      .qp_type = IBV_QPT_RC
    };

    int i;
    for(i = 0; i != num_qp; ++i) {
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

    for(int i = 0; i != num_qp; ++i) {
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
  for(int i = 0; i != num_qp; ++i) {
    if(i == rank)
      continue;
    ibv_destroy_qp(ctx->qps[i]);
  }

clean_srq:
  ibv_destroy_srq(ctx->srq);

clean_cq:
  ibv_destroy_cq(ctx->cq);

clean_mr:
  ibv_dereg_mr(ctx->msgs_mr);

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
  bbts_exch_t my_stuff = {
    .dest = my_dest,
    .addr = (uint64_t)(ctx->msgs + ctx->num_qp + qp_index),
    .rem_key = ctx->msgs_mr->rkey
  };
  bbts_exch_t rem_stuff;
  bool success = client_exch(servername, port, my_stuff, rem_stuff);
  if(!success) {
    return false;
  }
  rem_dest = rem_stuff.dest;
  ctx->msg_remote_addr[qp_index] = rem_stuff.addr;
  ctx->msg_remote_key[qp_index]  = rem_stuff.rem_key;

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
  bbts_exch_t my_stuff = {
    .dest = my_dest,
    .addr = (uint64_t)(ctx->msgs + ctx->num_qp + qp_index),
    .rem_key = ctx->msgs_mr->rkey
  };
  bbts_exch_t rem_stuff;
  bool success = server_exch(
        port,
        [&]()
        {
          return bbts_connect_context_helper(
                  ctx->qps[qp_index], ib_port, my_dest.psn, rem_stuff.dest);
        },
        my_stuff, rem_stuff);
  if(!success) {
    return false;
  }
  rem_dest = rem_stuff.dest;
  ctx->msg_remote_addr[qp_index] = rem_stuff.addr;
  ctx->msg_remote_key[qp_index]  = rem_stuff.rem_key;

  return true;
}

int bbts_connect_context(
    bbts_context_t *ctx,
    int32_t rank,
    std::vector<std::string> ips,
    uint8_t ib_port = 1, int port = 18515)
{
  std::vector<bbts_dest_t> rem_dests(ctx->num_qp);

  if (ibv_query_port(ctx->context, ib_port, &ctx->portinfo)) {
    return 1;
  }

  // Both the server and the client exch functions call bbts_connect_context_helper,
  // but they do so at different points.
  // This code was built off of an experiment which was built off of with libibverbs
  // pingpong example, and that is what they did.
  // It is unclear to me why that is the case.
  //
  for(int i = 0; i != ctx->num_qp; ++i) {
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
  for(int i = 0; i != ctx->num_qp; ++i) {
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
    rank(rank), opens(ips.size()), send_message_queue(ips.size()),
    current_recv_msg(0)
{
  int num_qp = ips.size();

  bbts_context_t *context = bbts_init_context(
    dev_name,
    rank,
    num_qp);

  if(!context) {
    throw std::runtime_error("make_connection: couldn't init context");
  }

  // post a receive before setting up a connection so that
  // when the connection is started, the receiving can occur
  int num_recvs = num_qp; // THE NUMBER OF RECVS == THE NUMBER OF RECV MESSAGES.
                          // THE which_recv INDEX HAS NOTHING TO DO WITH RANK, EVEN
                          // THOUGH IN THIS CASE
                          //   num_qp == NUMBER OF RANKS == NUMBER OF RECV MESSAGES
  // The number of recvs needs not be the number of queue pairs.
  for(int which_recv = 0; which_recv != num_recvs; ++which_recv) {
    bbts_post_recv(
      context->srq, 0,
      (void*)(context->msgs + which_recv),
      sizeof(bbts_message_t),
      context->msgs_mr->lkey,
      context->msgs_mr->rkey);
  }

  if(bbts_connect_context(context, rank, ips)) {
    throw std::runtime_error("make_connection: couldn't connect context");
  }

  destruct = false;
  std::fill(opens.begin(), opens.end(), 1);

  this->msgs_recv         = context->msgs;
  this->msgs_send         = context->msgs + num_qp;
  this->msgs_mr           = context->msgs_mr;
  this->msgs_remote_addr  = context->msg_remote_addr;
  this->msgs_remote_key   = context->msg_remote_key;

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
  ibv_dereg_mr(this->msgs_mr);
  delete[] msgs_recv;
  ibv_dealloc_pd(this->protection_domain);
  ibv_close_device(this->context);
}

std::future<bool> connection_t::send_bytes(int32_t dest_rank, tag_t send_tag, bytes_t bytes){
  if(dest_rank == rank) {
    throw std::invalid_argument("destination rank is same as current rank");
  }
  if(send_tag == 0) {
    throw std::invalid_argument("tag cannot be zero");
  }
  std::lock_guard<std::mutex> lk(send_m);
  send_init_queue.push_back({
    tag_rank_t{send_tag, dest_rank},
    std::unique_ptr<send_item_t>(new send_item_t(this, bytes, true))
  });
  return send_init_queue.back().second->get_future();
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
  return recv_init_queue.back().second->get_future();
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
  connection_t *connection, tag_t tag, int32_t dest_rank, uint64_t remote_addr, uint32_t remote_key)
{
  // TODO: what happens when size bigger than 2^31...
  // TODO: what wrid?
  bbts_post_rdma_write(
    connection->queue_pairs[dest_rank], tag,
    bytes.data, bytes.size, bytes_mr->lkey,
    remote_addr, remote_key);
}

bytes_t connection_t::recv_item_t::init(connection_t *connection, uint64_t size){
  bytes.data = (void*)(new char[size]);
  bytes.size = size;

  bytes_mr = ibv_reg_mr(
      connection->protection_domain,
      bytes.data, bytes.size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if(!bytes_mr) {
    throw std::runtime_error("couldn't register mr for recv bytes");
  }
  return bytes;
}

void connection_t::post_open_send(int32_t dest_rank, tag_t tag, uint64_t size, bool imm){
  send_message_queue[dest_rank].push({
    .type = bbts_message_t::message_type::open_send,
    .rank = dest_rank,
    .tag = tag,
    .m = {
      .open_send {
        .immediate = imm,
        .size = size
      }
    }
  });
}

void connection_t::post_open_recv(
  int32_t dest_rank, tag_t tag, uint64_t addr, uint64_t size, uint32_t key)
{
  send_message_queue[dest_rank].push({
    .type = bbts_message_t::message_type::open_recv,
    .rank = dest_rank,
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
  send_message_queue[dest_rank].push({
    .type = bbts_message_t::message_type::close_send,
    .rank = dest_rank,
    .tag = tag
  });
}

int connection_t::get_recv_rank(ibv_wc const& wc)
{
  for(int i = 0; i != opens.size(); ++i) {
    if(i == rank)
      continue;
    if(queue_pairs[i]->qp_num == wc.qp_num) {
      return i;
    }
  }
  throw std::runtime_error("could not get rank from work completion");
}

void connection_t::handle_message(int32_t recv_rank, bbts_message_t const& msg) {
  if(msg.type == bbts_message_t::message_type::open_send) {
    // send an open recv command. If the recv_item_t doesn't exist, create one.
    auto iter = recv_items.find(msg.tag);
    if(iter == recv_items.end()) {
      iter = recv_items.insert({
        msg.tag,
        std::unique_ptr<recv_item_t>(new recv_item_t(false))
      }).first;
    }
    // allocate and register memory
    recv_item_t& item = *(iter->second);
    item.init(this, msg.m.open_recv.size);

    // sending a message back to the location that sent this message
    this->post_open_recv(
      recv_rank, msg.tag,
      (uint64_t)item.bytes.data,
      item.bytes.size,
      item.bytes_mr->rkey);
  } else if(msg.type == bbts_message_t::message_type::open_recv) {
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
      item.pr.set_value(item.bytes);
      recv_items.erase(iter);
    } else {
      item.is_set = true;
    }
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
      // 3. attend to message queues
      // 4. receive a work request

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
          auto iter = recv_items.find(item.first);
          if(iter == recv_items.end()) {
            recv_items.insert(std::move(item));
          } else {
            // There are currently two recv items, but since
            // send_bytes and recv_bytes are only called once for each tag,
            // this means that the only relevant promise is the one from
            // the recv_bytes call in item--the one in recv_items is because
            // an OpenRecv command has already been sent.

            // If the bytes are alredy set,
            //   (1) set the promsie and
            //   (2) remove the entry in recv_items.
            // Otherwise the bytes are not set,
            //   move the relevant promise into recv_items
            if(iter->second->is_set) {
              item.second->pr.set_value(iter->second->bytes);
              recv_items.erase(iter);
            } else {
              iter->second->pr = std::move(item.second->pr);
            }
          }
        }
        recv_init_queue.resize(0);
      }

      // 3. actually send out messages
      for(int i = 0; i != opens.size(); ++i) {
        if(i == rank) {
          // there should never be anything here
          continue;
        }
        if(opens[i] && !send_message_queue[i].empty()) {
          msgs_send[i] = send_message_queue[i].front();
          send_message_queue[i].pop();
          bbts_post_send(
            queue_pairs[i], 0,
            (void*)(&msgs_send[i]), sizeof(bbts_message_t), msgs_mr->lkey,
            msgs_remote_addr[i], msgs_remote_key[i]);
          opens[i] = false;
        }
      }

      // 4. is there anything in the receive queue? If so, handle it
      int ne = ibv_poll_cq(completion_queue, 1, &work_completion);
      if(!ne) {
        continue;
      }
      if(work_completion.status) {
        throw std::runtime_error("work completion error");
      }
      bool is_send = work_completion.opcode == 0;
      bool is_recv = work_completion.opcode == 128;
      bool is_rdma_write = work_completion.opcode == 1;
      bool is_message = work_completion.wr_id == 0;
      int32_t wc_rank = get_recv_rank(work_completion);
      if(is_rdma_write && !is_message) {
        // an rdma write occured, inform destination we are done
        tag_t tag = work_completion.wr_id;
        post_close(wc_rank, tag);
      } else if(is_send && is_message) {
        // a send item is completed and the message is available again

        // But first, what is the message that is completed?
        // If our send_close made it, we are done with the send item.
        bbts_message_t const& msg = msgs_send[wc_rank];
        if(msg.type == bbts_message_t::message_type::close_send) {
          auto iter = send_items.find({msg.tag, wc_rank});
          if(iter == send_items.end()) {
            throw std::runtime_error("can't close send item, how can it not be here");
          }
          send_item_t& item = *(iter->second);
          item.pr.set_value(true);
          send_items.erase(iter);
        }

        opens[wc_rank] = true;
      } else if(is_recv && is_message) {
        // a message has been recvd
        // so post a recv back and handle the message

        // copy the message here so msg_recv can be used again
        // in the post recv
        bbts_message_t msg = msgs_recv[current_recv_msg];

        bbts_post_recv(
          shared_recv_queue, 0,
          (void*)(&msgs_recv[current_recv_msg]), sizeof(bbts_message_t),
          msgs_mr->lkey, msgs_mr->rkey);

        current_recv_msg++;
        if(current_recv_msg == opens.size()) {
          current_recv_msg = 0;
        }

        this->handle_message(wc_rank, msg);
      } else {
        throw std::runtime_error("unhandled item from recv queue");
      }
    }
  }
}

} // namespace ib
} // namespace bbts
