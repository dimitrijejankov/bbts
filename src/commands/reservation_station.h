#pragma once

#include <array>
#include <cassert>
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
#include "command_handler.h"
#include "command_handler_apply.h"
#include "command_handler_delete.h"
#include "command_handler_reduce.h"
#include "command_handler_move.h"
#include "heuristic.h"
#include "../tensor/tensor.h"
#include "../storage/storage.h"
#include "../utils/concurent_queue.h"

namespace bbts {

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

  // mark that a tensor was deleted
  bool retire_delete(tid_t id);

  // returns tensors that are scheduled to be remove from the storage
  tid_t get_to_delete();

  // get the next command, you must use the result of this command as it is a unique ptr
  [[nodiscard]] command_ptr_t get_next_command(command_t::op_type_t op_type);

  // register the tensor that was added externally,
  // that is it was not created through the execution of a command
  void register_tensor(tid_t _tid);

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
  void notify_ready_command(node_id_t node, const std::vector<command_t::command_tid_id_t> &tensors);

  // get the reduces that finished
  [[nodiscard]] std::vector<command_t::command_tid_id_t> commands_ready_for_node(node_id_t node, bool &is_done);

  // returns the rank
  [[nodiscard]] node_id_t get_rank() const;

 private:

  void _remove_tensor(tid_t in);

  void _tensor_became_available(bbts::tid_t tid);

  bool _is_done();

  void _check_if_finished();

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

  // the mutex
  std::mutex _m;

  // we use this to wait for commands
  std::condition_variable _cv;

  // a conditional variable that keeps track of how many local commands are left
  // it is signalled if we are done with all things this node needs to do
  std::condition_variable _done_cv;

  // is the node still running
  bool _shutdown = false;

  // is executing
  bool _is_executing = false;

  // tensors left to delete  
  size_t _tensors_left_to_delete = 0;

  // the rank 
  node_id_t _rank;

  // command handlers
  std::array<class std::shared_ptr<command_handler_t>, command_t::op_type_t::NUM_COMMANDS> _handlers;

  // keeps all the local tensors and information about them
  std::unordered_map<tid_t, internal_tensor_state_t> _tensors;

  // the local tensors commands are waiting fors
  std::unordered_multimap<tid_t, std::tuple<command_id_t, command_t::op_type_t>> _commands_waiting_for;

  // all the reduces we want to run
  std::shared_ptr<bbts::heuristic_t> _heuristic;

  // keeps track of all the reduces that have reduced to just one local value
  // these reduces must be communicated to the node that will initiate the distributed reduce
  std::vector<concurent_queue<bbts::command_t::command_tid_id_t>> _notify_done_reduces;

  // the tensors we want to delete from storage
  concurent_queue<tid_t> _to_delete;

  // mark them all as friends
  friend class ::bbts::command_handler_apply_t;
  friend class ::bbts::command_handler_reduce_t;
  friend class ::bbts::command_handler_delete_t;
  friend class ::bbts::command_handler_move_t;
};

using reservation_station_ptr_t = std::shared_ptr<reservation_station_t>;

}