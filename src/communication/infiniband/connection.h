#pragma once

#include <string>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>
#include <map>
#include <vector>
#include <queue>
#include <memory>
#include <iostream>

#include <infiniband/verbs.h>

#define _DCB_COUT_(x) // std::cout << x

namespace bbts {
namespace ib {

using std::future;
using std::promise;
using std::tuple;

using tag_t = uint64_t;
using ibv_qp_ptr = decltype(ibv_create_qp(NULL, NULL));

struct bytes_t {
  void* data;
  uint64_t size;
};

struct own_bytes_t {
  own_bytes_t():
    ptr(nullptr), size(0)
  {}

  own_bytes_t(bytes_t b):
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

using tag_rank_t = tuple<tag_t, int32_t>;

//struct tag_rank_t {
//  //tag_rank_t(tag_t tag, int rank): tag(tag), rank(rank) {}
//  tag_t tag;
//  int32_t rank;
//};
//
//struct tag_rank_less_t {
//  bool operator()(tag_rank_t const& lhs, tag_rank_t const& rhs) const {
//    if(lhs.tag == rhs.tag) {
//      return lhs.rank < rhs.rank;
//    }
//    return lhs.tag < rhs.tag;
//  }
//};

// As an algebraic data type, bbts_message_t looks something like this:
//  type BbtsMessage = (Rank, Tag, Info)
//  data Info =
//       OpenFromSend  Size
//       -- ^ tell the recv node how much memory it needs to recv the message
//     | OpenFromRecv  Addr Size Key
//       -- ^ tell the sending node where it can write to
//     | CloseFromSend
//       -- ^ tell the recv node the data has been written to
//     | FailFromSend
//       -- ^ tell the recv side data will not be written to
//     | FailFromRecv
//       -- ^ tell the send side there is no data to write to
struct bbts_message_t {
  enum message_type { open_send, open_recv, close_send, fail_send, fail_recv };

  message_type type;
  int32_t rank;
  int32_t from_rank;
  // ^ In theory, for any work completion, it's easy to get the from_rank.
  //   However, there was a bug where the from rank computed in this way was incorrect
  //   and it couldn't be determiend why.
  //   Instead, the from rank is also part of the message.
  //   TODO: remove from_rank.
  tag_t tag;
  union {
    struct {
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

struct virtual_send_queue_t;
struct virtual_recv_queue_t;

struct send_item_t;
struct recv_item_t;

using recv_item_ptr_t = std::shared_ptr<recv_item_t>;

// The connection_t object holds an "infinite" set of queues, each queue
// specified by a tag. The idea is that each queue must only be processing one
// item at a time.
struct connection_t {
  connection_t(
    std::string dev_name,
    int32_t rank,
    std::vector<std::string> ips);

  ~connection_t();

  future<bool> send(int32_t dest_rank, tag_t tag, bytes_t bytes);
  future<bool> send(int32_t dest_rank, tag_t tag, bytes_t bytes, ibv_mr* bytes_mr);

  future<tuple<bool, int32_t, own_bytes_t> > recv(tag_t tag);

  future<tuple<bool, int32_t> > recv_with_bytes(
    tag_t tag, bytes_t bytes);
  future<tuple<bool, int32_t > > recv_with_bytes(
    tag_t tag, bytes_t bytes, ibv_mr* bytes_mr);

  future<tuple<bool, own_bytes_t>> recv_from(int32_t from_rank, tag_t tag);
  future<bool> recv_from_with_bytes(
    int32_t from_rank, tag_t tag, bytes_t bytes);
  future<bool> recv_from_with_bytes(
    int32_t from_rank, tag_t tag, bytes_t bytes, ibv_mr* bytes_mr);

  int32_t get_rank() const { return rank; }
  int32_t get_num_nodes() const { return num_rank; }

  ibv_pd* get_protection_domain() { return protection_domain; }

private:
  void check_rank(int32_t other_rank) const {
    if(get_rank() == other_rank) {
      throw std::invalid_argument("cannot send and recv from self");
    }
  }
private:
  void post_send(int32_t dest_rank, bbts_message_t const& msg);

  // "post <do this> <from this>"
  //   post_fail_send comes from a send rank telling a recv rank a fail occured
  //   post_open_recv comes from a recv rank telling a send that a recv object is available
  //   and so on
  void post_open_send(int32_t dest_rank, tag_t tag, uint64_t size);
  void post_close_send(int32_t dest_rank, tag_t tag);
  void post_fail_send(int32_t dest_rank, tag_t tag);
  // only send side does this, recv side is not aware
  void post_rdma_write(int32_t dest_rank, bbts_rdma_write_t const& r);
  // the above post_x_send and post_rdma write are called inside this guy
  friend class virtual_send_queue_t;

  void post_open_recv(
    int32_t dest_rank, tag_t tag,
    uint64_t addr, uint64_t size, uint32_t key);
  void post_fail_recv(int32_t dest_rank, tag_t tag);
  // the boave post_x_recv are called inside this guy
  friend class virtual_recv_queue_t;

private:
  void poll();

  // these will make a virtual queue if there isn't already one at that tag, rank
  virtual_send_queue_t& get_send_queue(tag_t tag, int32_t dest_rank);
  virtual_recv_queue_t& get_recv_queue(tag_t tag, int32_t from_rank);

  void empty_send_init_queue();
  void empty_recv_init_queue();
  void empty_recv_anywhere_queue();
  void handle_work_completion(ibv_wc const& work_completion);

  // find out what queue pair was responsible for this work request
  int32_t get_recv_rank(ibv_wc const& wc) const;

private:
  int32_t rank;
  std::thread poll_thread;
  std::atomic<bool> destruct;

  // Things not yet handled.
  std::vector<tuple<tag_t, int32_t, send_item_t    > > send_init_queue;
  std::vector<tuple<tag_t, int32_t, recv_item_ptr_t> > recv_init_queue;
  std::vector<tuple<tag_t,          recv_item_ptr_t> > recv_anywhere_init_queue;
  // A mutext for each of the init queues
  std::mutex send_m, recv_m, recv_anywhere_m;

  // virtual send and recv queues
  // TODO: how do you go about deleting unusued queues?
  std::map<tag_rank_t, virtual_send_queue_t> virtual_send_queues;
  std::map<tag_rank_t, virtual_recv_queue_t> virtual_recv_queues;

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


