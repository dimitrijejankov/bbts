#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <memory>
#include <deque>
#include <utility>
#include <vector>
#include "command.h"
#include "../tensor/tensor.h"
#include "../storage/storage.h"
#include "../utils/concurent_queue.h"

namespace bbts {

// reduces (and partial reduces are given priority)
// if no reduces exist that we can run I pick an apply that is part of an reduce
// if not I pick an apply that will go into a reduce
// if not I pick the first apply available
class heuristic_t {
public:

  void execute() {
    std::unique_lock lk(m);
    is_executing = true;
    cv.notify_all();
  }

  void clear() {
    
    // clear all
    kernels.clear();
    reduces.clear();
    moves.clear();
  }

  bool queue_apply(command_ptr_t _command) {
    kernels.enqueue_copy(std::move(_command));
    return true;
  }

  bool  queue_partial_reduce(command_ptr_t _command) {
    kernels.enqueue_copy(std::move(_command));
    return true;
  }

  bool queue_reduce(command_ptr_t _command) {
    reduces.enqueue_copy(std::move(_command));
    return true;
  }

  bool queue_move(command_ptr_t _command) {
    moves.enqueue_copy(std::move(_command));
    return true;
  }

  bool next_apply(command_ptr_t &out) {

    {
      std::unique_lock lk(m);
      cv.wait(lk, [&]{return is_executing;});
    }
    return kernels.wait_dequeue(out);
  }

  bool next_reduce(command_ptr_t &out) {

    {
      std::unique_lock lk(m);
      cv.wait(lk, [&]{return is_executing;});
    }
    return reduces.wait_dequeue(out);
  }

  bool next_move(command_ptr_t &out) {

    {
      std::unique_lock lk(m);
      cv.wait(lk, [&]{return is_executing;});
    }
    return moves.wait_dequeue(out);
  }

  concurent_queue<command_ptr_t> kernels;

  concurent_queue<command_ptr_t> reduces;

  concurent_queue<command_ptr_t> moves;

  bool is_executing = false;

  std::mutex m;

  std::condition_variable cv;
};

// here we queue all the commands this node needs to execute that we need 
// to execute in conjunction with other nodes, are kept in the external_commands_queue_t
class reservation_station_t {
 public:

  reservation_station_t(node_id_t _node_id, int32_t num_nodes);

  // queue a command, this command has to be executed in the same thread and the commands
  // have to be queued in the exact order they are coming in
  bool queue_command(command_ptr_t _command);

  // mark that a command is processed
  bool retire_command(command_ptr_t _command);

  // get the next command, you must use the result of this command as it is a unique ptr
  [[nodiscard]] command_ptr_t get_next_move_command();
  [[nodiscard]] command_ptr_t get_next_kernel_command();
  [[nodiscard]] command_ptr_t get_distributed_reduce_command();

  // register the tensor that was added externally,
  // that is it was not created through the execution of a command
  void register_tensor(tid_t _tid);

  // returns tensors that are scheduled to be remove from the storage
  tid_t get_to_remove();

  // retire the remove command
  void retire_remove(tid_t _tid);

  // shutdown the reservation station
  void shutdown();

  // clear the reservation station
  void clear();

  // wait until all commands remote and local are executed
  void wait_until_finished();

  // execute all the scheduled commands
  void execute_scheduled_async();

  // stop executing all the commands
  void stop_executing();

  // notifies the reservation station that reduce commands of another node have completed
  // this node should then be able to kick off the remote reduce in case all nodes are ready
  void notify_ready_reduce(node_id_t node, const std::vector<command_id_t> &tensors);

  // get the reduces that finished
  [[nodiscard]] std::vector<tid_t> reduce_to_notify_node(node_id_t node, bool &is_done);

 private:

  bool _retire_remove(command_ptr_t _command);

  bool _retire_apply(command_ptr_t _command);

  bool _retire_reduce(command_ptr_t _command);

  bool _queue_delete_command(command_ptr_t _command);

  bool _queue_reduce_command(command_ptr_t _command);

  bool _queue_apply_command(command_ptr_t _command);

  bool _queue_move_command(command_ptr_t _command);

  void _remove_tensor(tid_t in);

  void _tensor_became_available(bbts::tid_t tid);

  // the state of the tensor
  struct internal_tensor_state_t {

    // the number of commands to read this, includes both the remote and local commands
    int32_t num_to_read = 0;

    // the number of commands to write this
    int32_t writing_tensor = false;

    // is the tensor created
    bool is_created = false;

    // is this tensor scheduled for delition
    bool scheduled_for_delition = false;
  };

  struct internal_reduce_state_t {

    // the inputs that need to be created here that the reduce is waiting for...
    std::vector<tid_t> missing_inputs;

    // the currently available inputs
    std::vector<tid_t> available_inputs;

    // we keep track of what nodes have finished all the local reduces they could they notify 
    // this node once they are done, this is kept for just the node that initiates the reduce
    std::vector<command_t::tid_node_id_t> waiting_for_nodes;
    std::vector<command_t::tid_node_id_t> done_nodes;

    command_ptr_t command;

    bool is_local = false;
  };

  void _update_reduce(internal_reduce_state_t &reduce);

  // the mutex
  std::mutex _m;

  // we use this to wait for commands
  std::condition_variable _cv;

  // the number of local commands to retire
  size_t _left_local_to_retire = 0;
  size_t _left_reduce_to_retire = 0;
  size_t _left_to_delete = 0;

  // a conditional variable that keeps track of how many local commands are left
  // it is signalled if we are done with all things this node needs to do
  std::condition_variable _done_cv;

  // is the node still running
  bool _shutdown = false;

  // is executing
  bool _is_executing = false;

  // the rank 
  node_id_t _rank;

  // the number of nodes
  size_t _num_nodes;

  // the id of the last command we have executed
  command_id_t _last_cmd = -1;

  // the local apply and move commands and the number of tensors they are waiting for
  std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

  // reduce commands we are keeping track of
  std::unordered_map<command_id_t, internal_reduce_state_t> reduce_commands;

  // keeps all the local tensors and information about them
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;

  // the local tensors commands are waiting for
  std::vector<std::unordered_multimap<tid_t, command_id_t>> _commands_waiting_for;

  // all the reduces we want to run
  heuristic_t _heuristic;

  // keeps track of all the reduces that have reduced to just one local value
  // these reduces must be communicated to the node that will initiate the distributed reduce
  std::vector<concurent_queue<command_id_t>> _notify_done_reduces;

  // deletion cv
  std::condition_variable _deletion_cv;

  // the tensors we want to delete from storage
  concurent_queue<tid_t> _to_delete;
};

using reservation_station_ptr_t = std::shared_ptr<reservation_station_t>;

}