#include "reorder_buffer.h" 
#include "command.h"
#include <cassert>

bbts::reorder_buffer_t::reorder_buffer_t() {}

void bbts::reorder_buffer_t::execute() {
  std::unique_lock lk(m);
  is_executing = true;
  cv.notify_all();
}

void bbts::reorder_buffer_t::stop_executing() {
  std::unique_lock lk(m);
  is_executing = false;
}

void bbts::reorder_buffer_t::shutdown() {

  // shutdown all the queues
  apply_queue.shutdown();
  reduce_queue.shutdown();
  move_queue.shutdown();
}

void bbts::reorder_buffer_t::clear() {

  // clear all the queues
  apply_queue.clear();
  reduce_queue.clear();
  move_queue.clear();
}

bool bbts::reorder_buffer_t::queue(command_ptr_t _command) {

  // check the type and send it to the right queue
  switch (_command->type) {
    case command_t::op_type_t::APPLY: apply_queue.enqueue_copy(std::move(_command)); break;
    case command_t::op_type_t::REDUCE: reduce_queue.enqueue_copy(std::move(_command)); break;
    case command_t::op_type_t::PARTIAL_REDUCE: apply_queue.enqueue_copy(std::move(_command)); break;
    case command_t::op_type_t::MOVE: move_queue.enqueue_copy(std::move(_command)); break;
    default: assert(false);
  }

  return true;
}

bool bbts::reorder_buffer_t::get_next(command_t::op_type_t type,
                                 command_ptr_t &out) {
  // check the type and send it to the right queue
  bool success = false;
  switch (type) {
    case command_t::op_type_t::APPLY: success = apply_queue.wait_dequeue(out); break;
    case command_t::op_type_t::REDUCE: success = reduce_queue.wait_dequeue(out); break;
    case command_t::op_type_t::PARTIAL_REDUCE: success = apply_queue.wait_dequeue(out); break;
    case command_t::op_type_t::MOVE: success = move_queue.wait_dequeue(out); break;
    default: assert(false);
  }

  // did we fail?
  if (!success) {
    return false;
  }

  // wait until we start executing
  std::unique_lock lk(m);
  cv.wait(lk, [&] { return is_executing; });
  return true;
}