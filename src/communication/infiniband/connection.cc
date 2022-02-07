#include "connection.h"
#include "utils.h"

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
  ibv_qp    *qp;
  ibv_port_attr portinfo;
  bbts_message_t *msgs;
  ibv_mr *msgs_mr;
  uint64_t msg_remote_addr;
  uint32_t msg_remote_key;
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
  std::cout << "POSTING RDMA WRITE[" << wr_id << "]: " << remote_key << std::endl;

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
  // // desinated initializors set everything else to zero,
  // // so this isn't necessary
  // // memset(&(wr.wr), 0, sizeof(wr.wr));
  // wr.wr.rdma.remote_addr = remote_addr;
  // wr.wr.rdma.rkey = remote_key;

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
  std::cout << "POSTING SEND[" << wr_id << "]: " << remote_key << std::endl;
  //std::cout << "---------------- POST SEND --------------------" << std::endl;
  //std::cout << "WR_ID: " << wr_id << std::endl;
  //std::cout << "REMOTE ADDR: " << remote_addr << std::endl;
  //std::cout << "REMOTE KEY:  " << remote_key  << std::endl;

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
  // // desinated initializors set everything else to zero,
  // // so this isn't necessary
  // // memset(&(wr.wr), 0, sizeof(wr.wr));
  // wr.wr.rdma.remote_addr = remote_addr;
  // wr.wr.rdma.rkey = remote_key;

  ibv_send_wr *bad_wr;

  if(ibv_post_send(qp, &wr, &bad_wr)) {
    throw std::runtime_error("ibv_post_send");
  }
}

void bbts_post_recv(
  ibv_qp *qp,
  uint64_t wr_id,
  void* local_addr, uint32_t local_size, uint32_t local_key,
  uint32_t expected_remote_key)
{
  std::cout << "POSTING RECV[" << wr_id << "]: " << expected_remote_key << std::endl;
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

  if(ibv_post_recv(qp, &wr, &bad_wr)) {
    throw std::runtime_error("Error in post recv");
  }
}

bbts_context_t* bbts_init_context(
    std::string dev_name,
    uint8_t ib_port = 1)
{
  bbts_context_t *ctx = new bbts_context_t;

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

  ctx->msgs = new bbts_message_t[2];
  ctx->msgs_mr = ibv_reg_mr(
      ctx->pd,
      ctx->msgs, 2*sizeof(bbts_message_t),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if(!ctx->msgs_mr) {
    goto clean_pd;
  }

  ctx->cq = ibv_create_cq(ctx->context, 8, NULL, NULL, 0);
  if (!ctx->cq) {
    goto clean_mr;
  }

  {
    ibv_qp_init_attr init_attr = {
      .send_cq = ctx->cq,
      .recv_cq = ctx->cq,
      .cap     = {
        .max_send_wr  = 4, // TODO: probably should be one, right?
        .max_recv_wr  = 4,
        .max_send_sge = 1,
        .max_recv_sge = 1
      },
      .qp_type = IBV_QPT_RC
    };

    ctx->qp = ibv_create_qp(ctx->pd, &init_attr);

    if (!ctx->qp)  {
      goto clean_cq;
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

    if (ibv_modify_qp(ctx->qp, &attr,
          IBV_QP_STATE              |
          IBV_QP_PKEY_INDEX         |
          IBV_QP_PORT               |
          IBV_QP_ACCESS_FLAGS)) {
      goto clean_qp;
    }
  }

  ibv_free_device_list(dev_list);
  return ctx;

clean_qp:
  ibv_destroy_qp(ctx->qp);

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

bool bbts_connect_context_helper(
      bbts_context_t *ctx,
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

	if (ibv_modify_qp(ctx->qp, &attr,
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
	if (ibv_modify_qp(ctx->qp, &attr,
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

bool bbts_client_exch_dest_and_connect(
  bbts_context_t *ctx,
  std::string servername,
  uint8_t ib_port, int port,
  bbts_dest_t const& my_dest,
  bbts_dest_t& rem_dest)
{
  bbts_exch_t my_stuff = {
    .dest = my_dest,
    .addr = (uint64_t)(ctx->msgs + 1),
    .rem_key = ctx->msgs_mr->rkey
  };
  bbts_exch_t rem_stuff;
  bool success = client_exch(servername, port, my_stuff, rem_stuff);
  if(!success) {
    return false;
  }
  rem_dest = rem_stuff.dest;
  ctx->msg_remote_addr = rem_stuff.addr;
  ctx->msg_remote_key  = rem_stuff.rem_key;

  if(!bbts_connect_context_helper(ctx, ib_port, my_dest.psn, rem_dest)) {
    return false;
  }

  return true;
}

bool bbts_server_exch_dest_and_connect(
  bbts_context_t *ctx,
  uint8_t ib_port, int port,
  bbts_dest_t const& my_dest,
  bbts_dest_t& rem_dest)
{
  bbts_exch_t my_stuff = {
    .dest = my_dest,
    .addr = (uint64_t)(ctx->msgs + 1),
    .rem_key = ctx->msgs_mr->rkey
  };
  bbts_exch_t rem_stuff;
  bool success = server_exch(
        port,
        [&]()
        {
          return bbts_connect_context_helper(ctx, ib_port, my_dest.psn, rem_stuff.dest);
        },
        my_stuff, rem_stuff);
  if(!success) {
    return false;
  }
  rem_dest = rem_stuff.dest;
  ctx->msg_remote_addr = rem_stuff.addr;
  ctx->msg_remote_key  = rem_stuff.rem_key;

  return true;
}

int bbts_connect_context(bbts_context_t *ctx, std::string ip_or_server,
    uint8_t ib_port = 1, int port = 18515)
{
  bool is_server = "server" == ip_or_server;

  bbts_dest_t my_dest;
  bbts_dest_t rem_dest;

  if (ibv_query_port(ctx->context, ib_port, &ctx->portinfo)) {
    return 1;
  }


  my_dest.lid = ctx->portinfo.lid;
  if (ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET && !my_dest.lid) {
    return 1;
  }

  my_dest.qpn = ctx->qp->qp_num;
  my_dest.psn = lrand48() & 0xffffff;

  // Both the server and the client exch functions also call
  // bbts_connect_context_helper, but they do so at different points in setting
  // up the connection. It is unclear to me what
  // 1. bbts_connect_context_helper is doing and should do, and
  // 2. why, if at all, they must call the function when they do.
  // This code was built off of an experiment which was built off of with libibverbs
  // pingpong example.
  bool success_exch =
    is_server
    ? bbts_server_exch_dest_and_connect(ctx,               ib_port, port, my_dest, rem_dest)
    : bbts_client_exch_dest_and_connect(ctx, ip_or_server, ib_port, port, my_dest, rem_dest);
  if(!success_exch){
    return 1;
  }

  // PRINT DEST INFO HERE
	printf("local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
	       my_dest.lid, my_dest.qpn, my_dest.psn);
	printf("remote address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
	       rem_dest.lid, rem_dest.qpn, rem_dest.psn);

  return 0;
}


Connection::Connection(
  std::string dev_name,
  std::string ip_or_server)
{
  bbts_context_t *context = bbts_init_context(dev_name);

  if(!context) {
    throw std::runtime_error("make_connection: couldn't init context");
  }

  // post a receive before setting up a connection so that
  // when the connection is started, the receiving can occur
  bbts_post_recv(
    context->qp, 0,
    (void*)context->msgs,
    sizeof(bbts_message_t),
    context->msgs_mr->lkey,
    context->msgs_mr->rkey);

  if(bbts_connect_context(context, ip_or_server)) {
    throw std::runtime_error("make_connection: couldn't connect context");
  }

  open = true;
  destruct = false;

  this->msg_recv         = context->msgs;
  this->msg_send         = context->msgs + 1;
  this->msgs_mr          = context->msgs_mr;
  this->msg_remote_addr  = context->msg_remote_addr;
  this->msg_remote_key   = context->msg_remote_key;

  this->context               = context->context;
  this->completion_queue      = context->cq;
  this->protection_domain     = context->pd;
  this->queue_pair            = context->qp;

  std::cout << "REMOTE ADDRESS "  << context->msg_remote_addr << std::endl;
  std::cout << "REMOTE KEY "      << context->msg_remote_key  << std::endl;
  std::cout << "LOCAL ADDRESS "   << ((uint64_t)context->msgs) << std::endl;
  std::cout << "PEER REMOTE KEY " << context->msgs_mr->rkey    << std::endl;

  delete context;

  poll_thread = std::thread(&Connection::poll, this);
}

Connection::~Connection() {
  destruct = true;
  poll_thread.join();

  ibv_destroy_qp(this->queue_pair);
  ibv_destroy_cq(this->completion_queue);
  ibv_dereg_mr(this->msgs_mr);
  delete[] msg_recv;
  ibv_dealloc_pd(this->protection_domain);
  ibv_close_device(this->context);
}

std::future<bool> Connection::send_bytes(tag_t send_tag, bytes_t bytes){
  if(send_tag == 0) {
    throw std::domain_error("tag cannot be zero");
  }
  std::lock_guard<std::mutex> lk(send_m);
  send_init_queue.push_back({
    send_tag,
    std::unique_ptr<SendItem>(new SendItem(this, bytes))
  });
  return send_init_queue.back().second->get_future();
}

std::future<bytes_t> Connection::recv_bytes(tag_t recv_tag){
  if(recv_tag == 0) {
    throw std::domain_error("tag cannot be zero");
  }
  recv_init_queue.push_back({
    recv_tag,
    std::unique_ptr<RecvItem>(new RecvItem(true))
  });
  return recv_init_queue.back().second->get_future();
}

Connection::SendItem::SendItem(Connection *connection, bytes_t b){
  bytes_mr = ibv_reg_mr(
      connection->protection_domain,
      bytes.data, bytes.size,
      IBV_ACCESS_LOCAL_WRITE);
  if(!bytes_mr) {
    throw std::runtime_error("couldn't register mr for bytes");
  }
}

void Connection::SendItem::send(
  Connection *connection, tag_t tag, uint64_t remote_addr, uint32_t remote_key)
{
  // TODO: what happens when size bigger than 2^31...
  // TODO: what wrid?
  bbts_post_rdma_write(connection->queue_pair, tag,
    bytes.data, bytes.size, bytes_mr->lkey,
    remote_addr, remote_key);
}

bytes_t Connection::RecvItem::init(Connection *connection, uint64_t size){
  if(bytes.data == nullptr || bytes.size < size) {
    bytes.data = (void*)(new char[size]);
    bytes.size = size;
  }
  bytes_mr = ibv_reg_mr(
      connection->protection_domain,
      bytes.data, bytes.size,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  return bytes;
}

void Connection::post_open_send(tag_t tag, uint64_t size){
  send_message_queue.push({
    .type = bbts_message_t::message_type::open_send,
    .tag = tag,
    .size = size
  });
}

void Connection::post_open_recv(tag_t tag, uint64_t addr, uint64_t size, uint32_t key){
  send_message_queue.push({
    .type = bbts_message_t::message_type::open_recv,
    .tag = tag,
    .addr = addr,
    .size = size,
    .key = key
  });
}

void Connection::post_close(tag_t tag){
  send_message_queue.push({
    .type = bbts_message_t::message_type::close_send,
    .tag = tag
  });
}

void Connection::poll(){
  ibv_wc work_completion;
  while(!destruct) {
    // It's not important that the thread catches the destruct exactly.
    for(int i = 0; i != 1000000; ++i) {
      // Empty the initialize queues
      {
        std::lock_guard<std::mutex> lk(send_m);
        for(auto && item: send_init_queue) {
          // does an rdma write already have to happen?
          auto iter = pending_sends.find(item.first);
          if(iter == pending_sends.end()) {
            // no? send an open send so you can get the remote info
            this->post_open_send(item.first, item.second->bytes.size);
            send_items.insert(std::move(item));
          } else {
            // yes? the remote info is already available, so do a remote write
            bbts_message_t& msg = iter->second;
            SendItem& send_item = *(item.second);
            send_item.send(this, msg.tag, msg.addr, msg.key);
            pending_sends.erase(iter);
            // still add it to send items so it can inform the peer when a
            // write has happened
            send_items.insert(std::move(item));
          }
        }
        send_init_queue.resize(0);
      }

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

      // Take something off of the message queue if open
      if(open && !send_message_queue.empty()) {
        *msg_send = send_message_queue.front();
        send_message_queue.pop();
        bbts_post_send(
          queue_pair, 0,
          (void*)msg_send, sizeof(bbts_message_t), msgs_mr->lkey,
          msg_remote_addr, msg_remote_key);
        open = false;
      }

      // poll for any events
      int ne = ibv_poll_cq(completion_queue, 1, &work_completion);
      if(!ne) {
        continue;
      }
      if(work_completion.status) {
        std::cout << "WORK COMPLETION STATUS " << work_completion.status << std::endl;
        throw std::runtime_error("work completion error");
      }

      bool is_send = work_completion.opcode == 0;
      bool is_rdma_write = work_completion.opcode == 1;
      bool is_message = work_completion.wr_id == 0;
      if(!is_message) {
        if(!is_rdma_write) {
          throw std::runtime_error("shouldn't happen");
        }

        // an rdma write occured
        tag_t tag = work_completion.wr_id;
        post_close(tag); // tell the peer we are done writing

        auto iter = send_items.find(tag);
        if(iter == send_items.end()) {
          throw std::runtime_error("this send item does not exist");
        }
        SendItem& item = *(iter->second);
        item.pr.set_value(true);
        send_items.erase(iter);
      //} else if(!is_send && !is_message) {
      //  throw std::runtime_error("how can you recv an rdma write?");
      } else if(is_send && is_message) {
        // a send item is completed and the message is available again
        open = true;
      } else if(!is_send && is_message) {
        // a message has been recvd
        // so post a recv back and handle the message

        // copy the message here
        bbts_message_t msg = *msg_recv;

        bbts_post_recv(
          queue_pair, 0,
          (void*)msg_recv, sizeof(bbts_message_t),
          msgs_mr->lkey, msgs_mr->rkey);

        // now handle the message
        if(msg.type == bbts_message_t::message_type::open_send) {
          // send an open recv command. If the RecvItem doesn't exist, create one.
          auto iter = recv_items.find(msg.tag);
          if(iter == recv_items.end()) {
            iter = recv_items.insert({
              msg.tag,
              std::unique_ptr<RecvItem>(new RecvItem(false))
            }).first;
          }
          // allocate and register memory
          RecvItem& item = *(iter->second);
          item.init(this, msg.size);

          post_open_recv(
            msg.tag,
            (uint64_t)item.bytes.data,
            item.bytes.size,
            item.bytes_mr->rkey);
        } else if(msg.type == bbts_message_t::message_type::open_recv) {
          // Do an rdma write if there is a send request available.
          // Otherwise, save the message for when send_bytes is called.
          auto iter = send_items.find(msg.tag);
          if(iter == send_items.end()) {
            pending_sends.insert({
              msg.tag,
              msg
            });
          } else {
            SendItem& item = *(iter->second);
            item.send(this, msg.tag, msg.addr, msg.key);
          }
        } else if(msg.type == bbts_message_t::message_type::close_send) {
          // the recv item knows it has been written to so set the future
          // and delete it UNLESS the promise is invalid
          auto iter = recv_items.find(msg.tag);
          if(iter == recv_items.end()) {
            throw std::runtime_error("invalid tag in close message");
          }
          RecvItem& item = *(iter->second);
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
    }
  }
}

} // namespace ib
} // namespace bbts
