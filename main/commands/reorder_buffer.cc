#include "reorder_buffer.h" 
#include "command.h"
#include <cassert>
#include <unordered_map>
#include <utility>

void bbts::apply_reorder_queue_t::queue(command_id_t apply_id) {

  auto it = _applies_into_reduce.find(apply_id);
  if(it == _applies_into_reduce.end()) {
    applies_not_into_reduces.insert(apply_id);
    return;
  }

  auto jt = _reduces_in_progress.find(it->second);
  if(jt == _reduces_in_progress.end()) {
    applies_into_reduces.insert(apply_id);
    return;
  }

  applies_into_ongoing_reduces.insert(apply_id);
}

void bbts::apply_reorder_queue_t::analyze(const std::vector<command_ptr_t> &cmds) {

  // this is an operational set to keep track of what tensors come from what apply
  std::unordered_map<tid_t, command_id_t> _apply_outputs;

  // go thorugh all the commands and init all the _applies_into_reduces and _reduce_to_applies
  for(const auto &cmd : cmds) {

    // if it is an apply store the output
    if (cmd->is_apply()) {

      for(size_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
        _apply_outputs[cmd->get_output(idx).tid] = cmd->id;
      }
    }
    else if(cmd->is_reduce()) {

      // we got a reduce go through all of its inputs and update the 
      for(size_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {
        
        // add the apply into reduce
        auto it = _apply_outputs.find(cmd->get_input(idx).tid);
        if(it != _apply_outputs.end()) {
          _applies_into_reduce.insert({ it->second, cmd->id });
          _reduce_to_applies.insert({ cmd->id, it->second });
        }
      }
    }
  }
}

void bbts::apply_reorder_queue_t::reduce_started(command_id_t reduce_id) {

  // mark that it is in progress
  if(_reduces_in_progress.find(reduce_id) != _reduces_in_progress.end()) {
    _reduces_in_progress.insert(reduce_id);
  }

  // go through all the affected applies
  auto range = _reduce_to_applies.equal_range(reduce_id);
  for (auto it = range.first; it != range.second; ++it) {

    // move the ongoing the apply from applies_into_reduces to applies_into_ongoing_reduces
    auto jt = applies_into_reduces.find(it->second);
    if(jt != applies_into_reduces.end()) {
      applies_into_ongoing_reduces.insert(it->second);
      applies_into_reduces.erase(jt);
    }
  }
}

void bbts::apply_reorder_queue_t::clear() {
  
  applies_not_into_reduces.clear();
  applies_into_reduces.clear();
  applies_into_ongoing_reduces.clear();
  _reduces_in_progress.clear();
  _applies_into_reduce.clear();
  _reduce_to_applies.clear();
}

bool bbts::apply_reorder_queue_t::has_any() {
  return !applies_not_into_reduces.empty() || 
         !applies_into_reduces.empty() || 
         !applies_into_ongoing_reduces.empty();
}

bbts::command_id_t bbts::apply_reorder_queue_t::get_next() {

  if(!applies_into_ongoing_reduces.empty()) {
    auto out = *applies_into_ongoing_reduces.begin();
    applies_into_ongoing_reduces.erase(applies_into_ongoing_reduces.begin());
    return out;
  }

  if(!applies_into_reduces.empty()) {
    auto out = *applies_into_reduces.begin();
    applies_into_reduces.erase(applies_into_reduces.begin());

    // we are now kicking off an apply that feeds directly into a reduce.. so we need to update stuff..
    auto it = _applies_into_reduce.find(out);
    assert(it != _applies_into_reduce.end());

    // mark that the reduce has started
    reduce_started(it->second);

    return out;
  }

  if(!applies_not_into_reduces.empty()) {
    auto out = *applies_not_into_reduces.begin();
    applies_not_into_reduces.erase(applies_not_into_reduces.begin());
    return out;
  }

  return -1;
}


bbts::reorder_buffer_t::reorder_buffer_t() {}

void bbts::reorder_buffer_t::analyze(const std::vector<command_ptr_t> &cmds) {
  std::unique_lock<std::mutex> lk(apply_reduce_m);
  apply_reorder_queue.analyze(cmds);
}

void bbts::reorder_buffer_t::execute() {
  std::unique_lock<std::mutex> lk(m);
  is_executing = true;
  cv.notify_all();
}

void bbts::reorder_buffer_t::stop_executing() {
  std::unique_lock<std::mutex> lk(m);
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
    
    partial_reduce_queue = {};
    apply_reorder_queue.clear();
    reduce_queue = {};
    move_queue = {};
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

void bbts::reorder_buffer_t::queue(command_ptr_t _command) {

  // check the type and send it to the right queue
  if(_command->type == command_t::op_type_t::APPLY) {
    _queue_apply(std::move(_command));
  }
  else if(_command->type == command_t::op_type_t::REDUCE) {
    _queue_reduce(std::move(_command));
  }
  else if(_command->type == command_t::op_type_t::PARTIAL_REDUCE) {
    _queue_partial_reduce(std::move(_command));
  }
  else if(_command->type == command_t::op_type_t::MOVE) {
    _queue_move(std::move(_command));
  } 
  else {
    assert(false);
  }
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
      
      // get the partial reduce
      out = std::move(partial_reduce_queue.front());
      partial_reduce_queue.pop();
    }
    else {

      // return the apply
      auto cmd_id = apply_reorder_queue.get_next();
      out = std::move(_applies[cmd_id]);
      _applies.erase(cmd_id);
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
  std::unique_lock<std::mutex> lk(m);
  cv.wait(lk, [&] { return is_executing; });
  return true;
}

void bbts::reorder_buffer_t::_queue_apply(command_ptr_t _command) {

  std::unique_lock<std::mutex> lk(apply_reduce_m);
  
  // insert the new apply
  apply_reorder_queue.queue(_command->id);
  _applies[_command->id] = std::move(_command);

  apply_reduce_cv.notify_all();
}

void bbts::reorder_buffer_t::_queue_reduce(command_ptr_t _command) {

  std::unique_lock<std::mutex> lk(dist_reduce_m);

  // queue a reduce
  reduce_queue.push(std::move(_command));
  dist_reduce_cv.notify_all();
}

void bbts::reorder_buffer_t::_queue_partial_reduce(command_ptr_t _command) {

  std::unique_lock<std::mutex> lk(apply_reduce_m);

  // add the partial reduce
  partial_reduce_queue.push(std::move(_command));
  apply_reduce_cv.notify_all();
}

void bbts::reorder_buffer_t::_queue_move(command_ptr_t _command) {

  std::unique_lock<std::mutex> lk(move_m);

  move_queue.push(std::move(_command));
  move_cv.notify_all();
}

bool bbts::reorder_buffer_t::any_applies_or_partial_reduces() {
  return !partial_reduce_queue.empty() || apply_reorder_queue.has_any();
}