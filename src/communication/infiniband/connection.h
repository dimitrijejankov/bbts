#pragma once

#include <string>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <vector>
#include <queue>

#include <infiniband/verbs.h>

namespace bbts {
namespace ib {

struct bytes_t {
  void* data;
  uint64_t size;
};

struct recv_bytes_t {
  recv_bytes_t(bytes_t b):
    ptr((char*)b.data), size(b.size)
  {}

  std::unique_ptr<char[]> ptr;
  uint64_t size;
};

template <typename T>
bytes_t to_bytes_t(T* ptr, size_t num_t_elems) {
  return {
    .data = (void*)ptr,
    .size = sizeof(T)*num_t_elems
  };
}

using tag_t = uint64_t;
using ibv_qp_ptr = decltype(ibv_create_qp(NULL, NULL));

struct tag_rank_t {
  //tag_rank_t(tag_t tag, int rank): tag(tag), rank(rank) {}
  tag_t tag;
  int32_t rank;
};

struct tag_rank_less_t {
  bool operator()(tag_rank_t const& lhs, tag_rank_t const& rhs) const {
    if(lhs.tag == rhs.tag) {
      return lhs.rank < rhs.rank;
    }
    return lhs.tag < rhs.tag;
  }
};

// As an algebraic data type, bbts_message_t looks something like this:
//   data BbtsMessage =
//       OpenSend  Rank Tag Immediate Size
//     | OpenRecv  Rank Tag Addr Size Key
//     | CloseSend Rank Tag
//     | FailSend  Rank Tag
// Once an OpenRecv is obtained, write the data into it.
// If an OpenRecv is not available, send an OpenSend command.
// Send CloseSend to remote data  holder whenever a write has finished
// FailSend tells send item it isn't gonna get recv data
struct bbts_message_t {
  enum message_type { open_send, open_recv, close_send, fail_send };
  message_type type;
  int32_t rank;
  int32_t from_rank;
  // ^ TODO: It'd be better to not also send from_rank, since in theory,
  // any work completion will know the qp_num, which means the from_rank can
  // be determined. Yet for some unknown reason, rank-from-qp_num is not
  // equalling from_rank in the some three-node experiment...
  // In the experiment, node 0 was getting a message from node 1 and ndoe 2,
  // and thinking msg from node 1 came from node 2 and msg from node 2 came
  // from node 1.
  //
  // TLDR: Including from_rank and using that when deciphering where a message
  // comes from prevents a bug...
  tag_t tag;
  union {
    struct {
      bool immediate;
      uint64_t size;
    } open_send;
    struct {
      uint64_t addr;
      uint64_t size;
      uint32_t key;
    } open_recv;
  } m;
};

struct bbts_rdma_write_t {
  uint64_t wr_id;
  void* local_addr;
  uint32_t local_size;
  uint32_t local_key;
  uint64_t remote_addr;
  uint32_t remote_key;
};

struct connection_t {
  connection_t(
    std::string dev_name,
    int32_t rank,
    std::vector<std::string> ips);

  ~connection_t();

  // ASSUMPTION: send_bytes and recv_bytes will be called at
  // most once for each tag value.
  // LOOSER ASSUMPTION: a tag can be used again when both sides
  //                    have completed the communication

  // Send these bytes. The memory can be released once the
  // returned future is ready. If the future returns a false,
  // the data was not recieved by the peer connection_t object.
  // This does not guarantee that recieve_bytes was called
  // by the peer connection, though.
  std::future<bool> send_bytes(int32_t dest_rank, tag_t send_tag, bytes_t bytes);
  std::future<recv_bytes_t> recv_bytes(tag_t recv_tag);

  // These functions are effectively the same as send_ and recv_ bytes except
  // the recving end will not allocate memory and will instead write into the provided
  // memory.
  // Case send_bytes, recv_bytes:
  //   S. tell R open send
  //   R. allocates memory, tells S.
  //   S. rdma writes to S.
  //   S. tell R close
  //   R. on close, bytes are written to, release memory in recv_bytes_t
  // Case send_bytes_wait, recv_bytes_wait:
  //   S. tell R open send with wait
  //   R. once user has called recv_bytes_wait, tell S
  //   S. rdma writes to S.
  //   S. tell R close
  //   R. on close, note success
  // (The ordering as shown is not necessarily what must happen)
  //
  // send_bytes/recv_bytes_wait and send_bytes_wait/recv_bytes pairs
  // are intended to be UNDEFINED.
  std::future<bool> send_bytes_wait(int32_t dest_rank, tag_t send_tag, bytes_t bytes);
  std::future<bool> recv_bytes_wait(tag_t recv_tag, bytes_t bytes);

  int32_t get_rank() const { return rank; }
  int32_t get_num_nodes() const { return num_rank; }

private:
  std::future<bool> send_bytes_(
    int32_t dest_rank, tag_t send_tag, bytes_t bytes, bool imm);
private:
  struct send_item_t {
    send_item_t(connection_t *connection, bytes_t b, bool imm);

    ~send_item_t() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    void send(
        connection_t *connection,
        tag_t tag, int32_t dest_rank,
        uint64_t remote_addr, uint32_t remote_key);

    std::future<bool> get_future(){ return pr.get_future(); }

    bytes_t bytes;
    ibv_mr *bytes_mr;
    std::promise<bool> pr;
    bool imm;
  };

  struct recv_item_t {
    recv_item_t(bool valid_promise):
      valid_promise(valid_promise),
      is_set(false),
      own(true),
      bytes_mr(nullptr)
    {}

    recv_item_t(bytes_t b):
      valid_promise(true),
      is_set(false),
      own(false),
      bytes(b)
    {}

    ~recv_item_t() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    void init(connection_t *connection, uint64_t size);

    std::future<recv_bytes_t> get_future_bytes();
    std::future<bool> get_future_complete();

    bool valid_promise;
    bool is_set;
    bool own;
    bytes_t bytes;
    ibv_mr *bytes_mr;
    std::promise<recv_bytes_t> pr_bytes;
    std::promise<bool> pr_complete;
  };

  using send_item_ptr_t = std::unique_ptr<send_item_t>;
  using recv_item_ptr_t = std::unique_ptr<recv_item_t>;

private:
  void post_send(int32_t dest_rank, bbts_message_t const& msg);
  void post_rdma_write(int32_t dest_rank, bbts_rdma_write_t const& r);
  void post_open_send(int32_t dest_rank, tag_t tag, uint64_t size, bool imm);
  void post_open_recv(
    int32_t dest_rank, tag_t tag,
    uint64_t addr, uint64_t size, uint32_t key);
  void post_close(int32_t dest_rank, tag_t tag);
  void post_fail_send(int32_t dest_rank, tag_t tag);
  void poll();

  // find out what queue pair was responsible for this work request
  int32_t get_recv_rank(ibv_wc const& wc);

  void handle_message(bbts_message_t const& msg);

private:
  int32_t rank;
  std::thread poll_thread;
  std::atomic<bool> destruct;

  // Things not yet handled.
  std::vector<std::pair<tag_rank_t, send_item_ptr_t> > send_init_queue;
  std::vector<std::pair<tag_t,      recv_item_ptr_t> > recv_init_queue;

  // Locations to write to but send bytes hasn't been called, so the
  // source data is not available.
  std::map<tag_rank_t, bbts_message_t, tag_rank_less_t> pending_sends;
  std::map<tag_t,      bbts_message_t                 > pending_recvs;

  std::map<tag_rank_t, send_item_ptr_t, tag_rank_less_t> send_items;
  std::map<tag_t,      recv_item_ptr_t                 > recv_items;

  std::mutex send_m, recv_m;

  std::queue<int> available_send_msgs;
  bbts_message_t* send_msgs;
  ibv_mr* send_msgs_mr;

  bbts_message_t* recv_msgs;
  ibv_mr* recv_msgs_mr;

  ibv_context *context;
  ibv_cq *completion_queue;
  ibv_pd *protection_domain;
  ibv_srq *shared_recv_queue;
  std::vector<ibv_qp_ptr> queue_pairs;

  // For each msgs_recv, there should be a recv open. current_recv_msg refers to the
  // msgs_recv index that has just been written to...
  // For exmaple:
  // - connection starts of mainting 1,...,num_recv recv msgs.
  // - the completion queue gets some recv items
  // - the first recv item should be written at msgs_recv[current_recv_msg].
  // - after processing that recv item and reposting msgs_recv[current_recv_msg],
  //   increment current_recv_msg.
  // - the next recv item processed from the completion queue should be written
  //   at msgs_recv[current_recv_msg].
  // - after processing that recv item and reposting it, increment current_recv_msg
  // And so on. except for the case where increment current_recv_msg will lead to
  // n+1. In that case, it should go back to zero.
  int current_recv_msg;

  int32_t num_rank;
  uint32_t num_recv;
  uint32_t num_send_per_qp;
  uint32_t num_write_per_qp;

  // for each rank except self, contain the number of send wrs left.
  // if this hits zero, you can't post a send to the given location
  std::vector<uint32_t> send_wr_cnts;
  std::vector<std::queue<bbts_message_t> > pending_msgs;

  // same same but for rdma writes
  std::vector<uint32_t> write_wr_cnts;
  std::vector<std::queue<bbts_rdma_write_t> > pending_writes;
};


} // namespace ib
} // namespace bbts


