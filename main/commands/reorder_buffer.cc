#include "reorder_buffer.h" 
#include "command.h"
#include <cassert>
#include <unordered_map>
#include <utility>

bbts::reorder_buffer_t::reorder_buffer_t() : 
  _apply_queue(std::set<command_id_t, reorder_buffer_cmp_t>(reorder_buffer_cmp_t(this))) {}

bool bbts::reorder_buffer_cmp_t::operator()(const command_id_t &lhs, const command_id_t &rhs) const {

  auto lhs_reduce = buffer->_applies_into_reduce.find(lhs);
  auto rhs_reduce = buffer->_applies_into_reduce.find(rhs);

  // if none of them go into an reduce sort them by the id
  if(lhs_reduce == buffer->_applies_into_reduce.end() && rhs_reduce == buffer->_applies_into_reduce.end()) {
    return lhs < rhs;
  }
  // if one of them does prefer that one
  else if(lhs_reduce != buffer->_applies_into_reduce.end() && rhs_reduce == buffer->_applies_into_reduce.end()) {
    return true;
  }
  else if(lhs_reduce == buffer->_applies_into_reduce.end() && rhs_reduce != buffer->_applies_into_reduce.end()) {
    return false;
  }

  // if both of them feed into a reduce pick the one that is feeding into a reduce that is in progress
  auto lhs_feeding = buffer->_reduces_in_progress.find(lhs_reduce->second);
  auto rhs_feeding = buffer->_reduces_in_progress.find(rhs_reduce->second);

  // if both of them feed into a reduce pick the one with the smallest command id
  if((lhs_feeding != buffer->_reduces_in_progress.end() && rhs_feeding != buffer->_reduces_in_progress.end())) {
    return *lhs_feeding < *rhs_feeding;
  }
  // if one of them does prefer that one
  else if(lhs_feeding != buffer->_reduces_in_progress.end() && rhs_feeding == buffer->_reduces_in_progress.end()) {
    return true;
  }
  else if(lhs_feeding == buffer->_reduces_in_progress.end() && rhs_feeding != buffer->_reduces_in_progress.end()) {
    return false;
  }

  // if none of them are in progress sort them by 
  return lhs < rhs;
}

void bbts::reorder_buffer_t::analyze(const std::vector<command_ptr_t> &cmds) {

  std::unique_lock<std::mutex> lk(apply_reduce_m);

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
    _apply_queue.clear();
    _apply_reduces.clear();
    _reduces_in_progress.clear();
    _applies_into_reduce.clear();
    _reduce_to_applies.clear();
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
      auto cmd_id = *_apply_queue.begin();
      out = std::move(_apply_reduces[cmd_id]);

      // remove the partial reduce
      _apply_queue.erase(_apply_queue.begin());
      _apply_reduces.erase(cmd_id);

      // mark that the reduce has started as we have calculated one of the inputs...
      auto it = _applies_into_reduce.find(cmd_id);
      if(it != _applies_into_reduce.end()) {
        _reduce_started(it->second);
      }
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

    // remove it since we are done with it
    _reduces_in_progress.erase(out->id);
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

void bbts::reorder_buffer_t::_reduce_started(bbts::command_id_t cmd_id) {

  auto it = _reduces_in_progress.find(cmd_id); 
  if(it == _reduces_in_progress.end()) {

    // remove all the affected applies
    std::vector<command_id_t> to_reinsert;
    auto range = _reduce_to_applies.equal_range(cmd_id);
    for (auto i = range.first; i != range.second; ++i) {
      auto jt = _apply_queue.find(i->second);
      if(jt != _apply_queue.end()) {
        _apply_queue.erase(jt);
        to_reinsert.push_back(i->second);
      }
    }

    // update the reduce in progress
    _reduces_in_progress.insert(cmd_id);

    // reinsert them back
    for (auto apply : to_reinsert) {
      _apply_queue.insert(apply);
    }
  }
}

void bbts::reorder_buffer_t::_partial_reduce_ended(bbts::command_id_t cmd_id) {

  std::unique_lock<std::mutex> lk(apply_reduce_m);
  
  // remove these as they are no longer necessary
  auto range = _reduce_to_applies.equal_range(cmd_id);
  for (auto i = range.first; i != range.second; ++i) {
    _applies_into_reduce.erase(i->second);
  }

  // remove the reduce in progress as with the final reduce scheduled 
  // we no longer need to keep track of this
  auto it = _reduces_in_progress.find(cmd_id); 
  if(it != _reduces_in_progress.end()) {
    _reduces_in_progress.erase(it);
  }
}

void bbts::reorder_buffer_t::_queue_apply(command_ptr_t _command) {

  std::unique_lock<std::mutex> lk(apply_reduce_m);
  
  // insert the new apply
  _apply_queue.insert(_command->id);
  _apply_reduces[_command->id] = std::move(_command);

  apply_reduce_cv.notify_all();
}

void bbts::reorder_buffer_t::_queue_reduce(command_ptr_t _command) {

  // signal that the final distributed reduce started
  _partial_reduce_ended(_command->id);

  std::unique_lock<std::mutex> lk(dist_reduce_m);

  // queue a reduce
  reduce_queue.push(std::move(_command));
  dist_reduce_cv.notify_all();
}

void bbts::reorder_buffer_t::_queue_partial_reduce(command_ptr_t _command) {
  if(_command->get_output(0).tid > 0) {
    _partial_reduce_ended(_command->id);
  }
  else {
     // signal that the reduce is started
     std::unique_lock<std::mutex> lk(apply_reduce_m);
    _reduce_started(_command->id);
  }

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
  return !partial_reduce_queue.empty() || !_apply_queue.empty();
}