#include "reservation_station.h"
#include "../server/static_config.h"
#include "command.h"
#include "command_handlers/command_handler.h"
#include "command_handlers/command_handler_apply.h"
#include "command_handlers/command_handler_delete.h"
#include "command_handlers/command_handler_move.h"
#include "command_handlers/command_handler_reduce.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

bbts::reservation_station_t::reservation_station_t(bbts::node_id_t _node_id, int32_t num_nodes) : _rank(_node_id),
                                                                                                  _notify_done_reduces(num_nodes) {

  _handlers[command_t::op_type_t::APPLY] = std::static_pointer_cast<command_handler_t>(std::make_shared<command_handler_apply_t>(this));
  _handlers[command_t::op_type_t::MOVE] = std::static_pointer_cast<command_handler_t>(std::make_shared<command_handler_move_t>(this));
  _handlers[command_t::op_type_t::REDUCE] = std::static_pointer_cast<command_handler_t>(std::make_shared<command_handler_reduce_t>(this));
  _handlers[command_t::op_type_t::PARTIAL_REDUCE] = _handlers[command_t::op_type_t::REDUCE];
  _handlers[command_t::op_type_t::DELETE] = std::static_pointer_cast<command_handler_t>(std::make_shared<command_handler_delete_t>(this));
}

void bbts::reservation_station_t::queue_commands(const std::vector<command_ptr_t> &cmds) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  // analyze all the commands
  _reorder_buffer.analyze(cmds);

  // schedule them all at once
  for (auto &_cmd : cmds) {

    // if it uses the node
    if (_cmd->uses_node(_rank)) {
      _handlers[_cmd->type]->schedule_command(_cmd->clone());
    }
  }
}

bool bbts::reservation_station_t::retire_command(command_ptr_t _command) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);
  return _handlers[_command->type]->retire_command(std::move(_command));
}

bool bbts::reservation_station_t::retire_delete(bbts::tid_t id) {

  std::unique_lock<std::mutex> lk(_m);

  // we got one less
  _tensors_left_to_delete--;
  _check_if_finished();

  // we succeeded
  return true;
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_command(command_t::op_type_t op_type) {
  
  command_ptr_t out;
  if(!_reorder_buffer.get_next(op_type, out)) {
    return nullptr;
  }
  return std::move(out);
}

void bbts::reservation_station_t::register_tensor(tid_t _tid) {

  // lock the tensor
  std::unique_lock<std::mutex> lk(_m);

  // get the tensor state if any
  auto &s = _tensors[_tid];

  // make sure that it was not created before
  assert(!s.is_created);

  // we are done writing to the tensor and
  s.is_created = true;
  s.writing_tensor = false;

  // go through the commands that are waiting
  _tensor_became_available(_tid);
}

bbts::tid_t bbts::reservation_station_t::get_to_delete() {

  // wait until we have something to do
  // if the queue is shutdown just return -1
  tid_t t;
  bool success = _to_delete.wait_dequeue(t);
  return success ? t : -1;
}

void bbts::reservation_station_t::shutdown() {

  // set the flag
  std::unique_lock<std::mutex> lk(_m);

  _shutdown = true;

  // notify that we are done
  _cv.notify_all();
  _to_delete.shutdown();
  _reorder_buffer.shutdown();
  for(auto &ndr : _notify_done_reduces) { ndr.shutdown(); }
}

void bbts::reservation_station_t::clear() {

  std::unique_lock<std::mutex> lk(_m);

  _reorder_buffer.clear();
  _commands_waiting_for.clear();
  _tensors.clear();

  _handlers[command_t::op_type_t::APPLY]->clear();
  _handlers[command_t::op_type_t::MOVE]->clear();
  _handlers[command_t::op_type_t::REDUCE]->clear();
  _handlers[command_t::op_type_t::DELETE]->clear();

  for(auto &t : _notify_done_reduces) {
    t.clear();
  }

  _to_delete.clear();
}

void bbts::reservation_station_t::wait_until_finished() {

  // wait until all the commands are run
  std::unique_lock<std::mutex> lk(_m);
  _done_cv.wait(lk, [&]{

    // check if all handlers have finished their commands
    return _handlers[command_t::op_type_t::APPLY]->is_done() &&
           _handlers[command_t::op_type_t::MOVE]->is_done() &&
           _handlers[command_t::op_type_t::REDUCE]->is_done() &&
           _handlers[command_t::op_type_t::DELETE]->is_done();
  });
}

void bbts::reservation_station_t::execute_scheduled_async() {

  // kick off everything
  std::unique_lock<std::mutex> lk(_m);
  _reorder_buffer.execute();
  _is_executing = true;
  _cv.notify_all();
}

void bbts::reservation_station_t::stop_executing() {

  // update the flag
  std::unique_lock<std::mutex> lk(_m);
  _reorder_buffer.stop_executing();
  _is_executing = false;
}

void bbts::reservation_station_t::notify_ready_command(node_id_t node, const std::vector<command_t::command_tid_id_t> &commands) {

  // lock the tensor
  std::unique_lock<std::mutex> lk(_m);

  _handlers[command_t::op_type_t::APPLY]->commands_finished_on_node(commands, node);
  _handlers[command_t::op_type_t::MOVE]->commands_finished_on_node(commands, node);
  _handlers[command_t::op_type_t::REDUCE]->commands_finished_on_node(commands, node);
  _handlers[command_t::op_type_t::DELETE]->commands_finished_on_node(commands, node);
}

[[nodiscard]] std::vector<bbts::command_t::command_tid_id_t> bbts::reservation_station_t::commands_ready_for_node(node_id_t node, bool &is_done) {
  std::vector<bbts::command_t::command_tid_id_t> commands;
  is_done = !_notify_done_reduces[node].wait_dequeue_all(commands);
  return std::move(commands);
}

[[nodiscard]] bbts::node_id_t bbts::reservation_station_t::get_rank() const {
  return _rank;
}

void bbts::reservation_station_t::_remove_tensor(tid_t tid) {

  // remove the tensor from the storage
  _to_delete.enqueue(tid);

  // remove the tensor
  _tensors.erase(tid);
}

void bbts::reservation_station_t::_tensor_became_available(bbts::tid_t tid) {

  // go through the commands that are waiting
  auto cw = _commands_waiting_for.equal_range(tid);
  for (auto it = cw.first; it != cw.second;) {

    // signal that the tensors become available
    auto [command_id, type] = it->second;
    _handlers[type]->tensor_available(command_id, tid);

    // remove the command from the waiting list
    it = _commands_waiting_for.erase(it);
  }
}

bool bbts::reservation_station_t::_is_done() {

  // check if the reservation station is done
  return _handlers[command_t::op_type_t::APPLY]->is_done() &&
         _handlers[command_t::op_type_t::MOVE]->is_done() &&
         _handlers[command_t::op_type_t::REDUCE]->is_done() &&
         _handlers[command_t::op_type_t::DELETE]->is_done();
}

void bbts::reservation_station_t::_check_if_finished() {
  if(_is_done()) {
    _done_cv.notify_all();
  }
}