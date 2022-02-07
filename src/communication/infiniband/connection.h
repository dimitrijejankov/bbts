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

// As an algebraic data type, bbts_message_t looks something like this:
//   data BbtsMessage =
//       OpenSend Size Tag
//     | OpenRecv Addr Size Key Tag
//     | CloseSend Tag
// Once an OpenRecv is obtained, write the data into it.
// If an OpenRecv is not available, send an OpenSend command.
// Send CloseSend to remote data  holder whenever a write has finished
struct bbts_message_t {
  enum message_type { open_send, open_recv, close_send };
  message_type type;
  tag_t tag;
  uint64_t addr;
  uint64_t size;
  uint32_t key;
};

struct Connection {
  Connection(
    std::string dev_name,
    std::string ip_or_server);

  ~Connection();

  // ASSUMPTION: send_bytes and recv_bytes will be called at
  // most once for each tag value.

  // Send these bytes. The memory can be released once the
  // returned future is ready. If the future returns a false,
  // the data was not recieved by the peer Connection object.
  // This does not guarantee that recieve_bytes was called
  // by the peer connection, though.

  std::future<bool> send_bytes(tag_t send_tag, bytes_t bytes);
  std::future<bytes_t> recv_bytes(tag_t recv_tag);
private:
  struct SendItem {
    SendItem(Connection *connection, bytes_t b);

    ~SendItem() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    void send(Connection *connection, tag_t tag, uint64_t remote_addr, uint32_t remote_key);

    std::future<bool> get_future(){ return pr.get_future(); }

    bytes_t bytes;
    ibv_mr *bytes_mr;
    std::promise<bool> pr;
  };

  struct RecvItem {
    RecvItem(bool valid_promise): is_set(false), valid_promise(valid_promise){}

    ~RecvItem() {
      if(bytes_mr) {
        ibv_dereg_mr(bytes_mr);
      }
    }

    bytes_t init(Connection *connection, uint64_t size);

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

  using SendItemPtr = std::unique_ptr<SendItem>;
  using RecvItemPtr = std::unique_ptr<RecvItem>;

private:
  void post_open_send(tag_t tag, uint64_t size);
  void post_open_recv(tag_t tag, uint64_t addr, uint64_t size, uint32_t key);
  void post_close(tag_t tag);
  void poll();

private:
  std::thread poll_thread;
  std::atomic<bool> destruct;

  std::queue<bbts_message_t> send_message_queue;
  bool open;

  std::vector<std::pair<tag_t, SendItemPtr> > send_init_queue;
  std::vector<std::pair<tag_t, RecvItemPtr> > recv_init_queue;

  std::map<tag_t, bbts_message_t> pending_sends;

  std::map<tag_t, SendItemPtr> send_items;
  std::map<tag_t, RecvItemPtr> recv_items;

  std::mutex send_m, recv_m;

  // msg_recv and msg_send live in consecutive memory, both managed
  // by msgs_mr
  bbts_message_t* msg_recv;
  bbts_message_t* msg_send;
  ibv_mr* msgs_mr;
  uint64_t msg_remote_addr;
  uint32_t msg_remote_key;

  ibv_context *context;
  ibv_cq *completion_queue;
  ibv_pd *protection_domain;
  ibv_qp *queue_pair;
};


} // namespace ib
} // namespace bbts
