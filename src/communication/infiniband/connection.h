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

using tag_t = uint64_t;
using ibv_qp_ptr = decltype(ibv_create_qp(NULL, NULL));

struct tag_rank_t {
  //tag_rank_t(tag_t tag, int rank): tag(tag), rank(rank) {}
  tag_t tag;
  int rank;
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
//       OpenSend  Rank Tag Size
//     | OpenRecv  Rank Tag Addr Size Key
//     | CloseSend Rank Tag
// Once an OpenRecv is obtained, write the data into it.
// If an OpenRecv is not available, send an OpenSend command.
// Send CloseSend to remote data  holder whenever a write has finished
struct bbts_message_t {
  enum message_type { open_send, open_recv, close_send };
  message_type type;
  int rank;
  tag_t tag;
  uint64_t addr;
  uint64_t size;
  uint32_t key;
};

struct connection_t {
  connection_t(
    std::string dev_name,
    int rank,
    std::vector<std::string> ips);

  ~connection_t();

  // ASSUMPTION: send_bytes and recv_bytes will be called at
  // most once for each tag value.

  // Send these bytes. The memory can be released once the
  // returned future is ready. If the future returns a false,
  // the data was not recieved by the peer connection_t object.
  // This does not guarantee that recieve_bytes was called
  // by the peer connection, though.
  std::future<bool> send_bytes(int dest_rank, tag_t send_tag, bytes_t bytes);
  std::future<bytes_t> recv_bytes(tag_t recv_tag);
private:
  struct send_item_t {
    send_item_t(connection_t *connection, bytes_t b);

    ~send_item_t() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    void send(
        connection_t *connection,
        tag_t tag, int dest_rank,
        uint64_t remote_addr, uint32_t remote_key);

    std::future<bool> get_future(){ return pr.get_future(); }

    bytes_t bytes;
    ibv_mr *bytes_mr;
    std::promise<bool> pr;
  };

  struct recv_item_t {
    recv_item_t(bool valid_promise): is_set(false), valid_promise(valid_promise){}

    ~recv_item_t() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    bytes_t init(connection_t *connection, uint64_t size);

    std::future<bytes_t> get_future(){
      if(valid_promise) {
        return pr.get_future();
      } else {
        throw std::runtime_error("invalid promise in recv item");
      }
    }

    bool valid_promise;
    bool is_set;
    bytes_t bytes;
    ibv_mr *bytes_mr;
    std::promise<bytes_t> pr;
  };

  using send_item_ptr_t = std::unique_ptr<send_item_t>;
  using recv_item_ptr_t = std::unique_ptr<recv_item_t>;

private:
  void post_open_send(int dest_rank, tag_t tag, uint64_t size);
  void post_open_recv(
    int dest_rank, tag_t tag,
    uint64_t addr, uint64_t size, uint32_t key);
  void post_close(int dest_rank, tag_t tag);
  void poll();

  // find out what queue pair was responsible for this work request
  int get_recv_rank(ibv_wc const& wc);

  void handle_message(int recv_rank, bbts_message_t const& msg);

private:
  int rank;
  std::thread poll_thread;
  std::atomic<bool> destruct;

  // The polling queue will take from this and post a send message to the
  // appropriate queue.
  std::vector<std::queue<bbts_message_t> > send_message_queue;
  // Whenever an open[rank] flag is set, an item from send_message_queue[rank]
  // can be posted.
  std::vector<char> opens;

  // Things not yet moved to the send_message_queue.
  std::vector<std::pair<tag_rank_t, send_item_ptr_t> > send_init_queue;
  std::vector<std::pair<tag_t,      recv_item_ptr_t> > recv_init_queue;

  // Locations to write to but send bytes hasn't been called, so the
  // source data is not available.
  std::map<tag_rank_t, bbts_message_t, tag_rank_less_t> pending_sends;

  std::map<tag_rank_t, send_item_ptr_t, tag_rank_less_t> send_items;
  std::map<tag_t,      recv_item_ptr_t                 > recv_items;

  std::mutex send_m, recv_m;

  // msgs_recv and msgs_send live in consecutive memory, both managed
  // by msgs_mr
  bbts_message_t* msgs_recv;
  bbts_message_t* msgs_send;
  ibv_mr* msgs_mr;
  std::vector<uint64_t> msgs_remote_addr;
  std::vector<uint32_t> msgs_remote_key;

  ibv_context *context;
  ibv_cq *completion_queue;
  ibv_pd *protection_domain;
  ibv_srq *shared_recv_queue;
  std::vector<ibv_qp_ptr> queue_pairs;
};


} // namespace ib
} // namespace bbts


