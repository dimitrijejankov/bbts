#include "reorder_buffer.h" 
#include "command.h"
#include <cassert>
#include <utility>

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
  _shutdown = true;
  apply_reduce_cv.notify_all();
  dist_reduce_cv.notify_all();
  move_cv.notify_all();
}

void bbts::reorder_buffer_t::clear() {

  // clear the apply reduce queue
  {
    std::unique_lock<std::mutex> lk(apply_reduce_m);
    apply_queue = {};
    partial_reduce_queue = {};
  }

  // clear the move queue
  {
    std::unique_lock<std::mutex> lk(move_m);
    move_queue = {};
  }

  // clear the distributed reduce queue
  {
    std::unique_lock<std::mutex> lk(dist_reduce_m);
    reduce_queue = {};
  }
}

bool bbts::reorder_buffer_t::queue(command_ptr_t _command) {

  // check the type and send it to the right queue
  if(_command->type == command_t::op_type_t::APPLY) {
    std::unique_lock<std::mutex> lk(apply_reduce_m);
    apply_queue.push(std::move(_command));
    apply_reduce_cv.notify_all();
  }
  else if(_command->type == command_t::op_type_t::REDUCE) {
    std::unique_lock<std::mutex> lk(dist_reduce_m);
    reduce_queue.push(std::move(_command));
    dist_reduce_cv.notify_all();
  }
  else if(_command->type == command_t::op_type_t::PARTIAL_REDUCE) {
    std::unique_lock<std::mutex> lk(apply_reduce_m);
    partial_reduce_queue.push(std::move(_command));
    apply_reduce_cv.notify_all();
  }
  else if(_command->type == command_t::op_type_t::MOVE) {
    std::unique_lock<std::mutex> lk(move_m);
    move_queue.push(std::move(_command));
    move_cv.notify_all();
  }
  else {
    assert(false);
  }

  return true;
}

bool bbts::reorder_buffer_t::get_next(command_t::op_type_t type, command_ptr_t &out) {

  if(type == command_t::op_type_t::APPLY) {

    // wait until we have something
    std::unique_lock<std::mutex> lk(apply_reduce_m);
    apply_reduce_cv.wait(lk, [&]{return _shutdown || any_applies_or_partial_reduces();});

    // should we shutdown
    if(_shutdown) {
      return false;
    }

    // check if there are any partial reduces we give them priority
    if(!partial_reduce_queue.empty()) {
      out = std::move(partial_reduce_queue.front());
      partial_reduce_queue.pop();
    }
    else {

      // return the apply
      out = std::move(apply_queue.front());
      apply_queue.pop();
    }
  }
  else if(type == command_t::op_type_t::REDUCE) {
    
    // wait until we have something
    std::unique_lock<std::mutex> lk(dist_reduce_m);
    dist_reduce_cv.wait(lk, [&]{return _shutdown || !reduce_queue.empty();});

    // should we shutdown
    if(_shutdown) {
      return false;
    }

    // return the reduce
    out = std::move(reduce_queue.front());
    reduce_queue.pop();
  }
  else if (type == command_t::op_type_t::MOVE) {

    // wait until we have something
    std::unique_lock<std::mutex> lk(move_m);
    move_cv.wait(lk, [&]{return _shutdown || !move_queue.empty();});

    // should we shutdown
    if(_shutdown) {
      return false;
    }

    // return the move
    out = std::move(move_queue.front());
    move_queue.pop();
  }

  // wait until we start executing
  std::unique_lock lk(m);
  cv.wait(lk, [&] { return is_executing; });
  return true;
}

bool bbts::reorder_buffer_t::any_applies_or_partial_reduces() {
  return !partial_reduce_queue.empty() || !apply_queue.empty();
}