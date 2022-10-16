#include "command_handler_reduce.h"
#include "../command.h"
#include "../reservation_station.h"

namespace bbts {

bool command_handler_reduce_t::retire_command(command_ptr_t _command) {

  // partial reduce has an anonymous tensor as output
  if (_command->get_output(0).tid < 0) {
    
    // go through all the inputs and remove the ones we do not need
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto in = _command->get_input(i);

      // if this tensor is not on our node we don't do anything
      if(in.node != _rs->_rank) {
        continue;
      }

      // get the tid
      auto tid = in.tid;
      auto &s = _rs->_tensors[tid];

      // decrement the number of readers
      s.num_to_read--;
      assert(s.num_to_read >= 0);

      // if there are no command that is writing to this tensor
      // reading this tensor and the tensor is scheduled for deletion, delete it here
      if (s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

        // remove the tensor immediately
        _rs->_remove_tensor(in.tid);
      }
    }

    // parital reduce
    auto &reduce = reduce_commands[_command->id];
    
    // store the new input
    reduce.available_inputs.push_back(_command->get_outputs()[0].tid);
    
    // remove the -1 from the missing inputs
    auto it = std::find(reduce.missing_inputs.begin(), reduce.missing_inputs.end(), -1);
    assert(it != reduce.missing_inputs.end());
    *it = *reduce.missing_inputs.rbegin();
    reduce.missing_inputs.pop_back();

    // add the tensor
    auto &t = _rs->_tensors[_command->get_outputs()[0].tid];
    t.writing_tensor = false;
    t.is_created = true;
    t.num_to_read = 1;
    t.scheduled_for_delition = true;

    // update the reduce
    _update_reduce(reduce);
  }
  else {

    // get the tensor required in the output and update stuff if necessary
    auto out = _command->get_output(0);
    if(out.node == _rs->_rank) {

      // get the tid
      auto tid = out.tid;
      auto &s = _rs->_tensors[tid];

      // make sure that it was not created before
      assert(!s.is_created);

      // we are done writing to the tensor and
      s.is_created = true;
      s.writing_tensor = false;

      // remove the tensor if it is not needed
      if (s.num_to_read == 0 && s.scheduled_for_delition) {

        // remove the tensor immediately
        _rs->_remove_tensor(out.tid);
      }

      // go through the commands that are waiting
      _rs->_tensor_became_available(out.tid);
    }

    // check if we need to remove 
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto in = _command->get_input(i);

      // if this tensor is not on our node we don't do anything
      if(in.node != _rs->_rank) {
        continue;
      }

      // get the tid
      auto tid = in.tid;
      auto &s = _rs->_tensors[tid];

      // decrement the number of readers
      s.num_to_read--;
      assert(s.num_to_read >= 0);

      // if there are no command that is writing to this tensor
      // reading this tensor and the tensor is scheduled for deletion, delete it here
      if (s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

        // remove the tensor immediately
        _rs->_remove_tensor(in.tid);
      }
    }

    // remove the command
    reduce_commands.erase(_command->id);
  
    // we have one less to retire
    _left_reduce_to_retire--;
    _rs->_check_if_finished();
  }

  return true;
}

bool command_handler_reduce_t::schedule_command(command_ptr_t _command) {

  auto &reduce = reduce_commands[_command->id];
  auto root_node = _command->get_root_node();

  // go through all the inputs
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // check if this is a remote tensor or a local tensor
    if(_in.node == _rs->_rank) {

      // if it is a local we need to check if it was created already
      auto &s = _rs->_tensors[_in.tid];

      // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
      if(s.scheduled_for_delition) { return false; }

      // we are reading this tensor
      s.num_to_read++;

      // if it was not created we need to keep track of that
      if(!s.is_created) {

        // mark that this command is waiting
        _rs->_commands_waiting_for.insert({_in.tid, { _command->id, command_t::op_type_t::REDUCE }});
        reduce.missing_inputs.push_back(_in.tid);
      }
      else {

        // the tensor is available so we add it here
        reduce.available_inputs.push_back(_in.tid);
      }
    }
    // if it is not a local tensor check if we are the node responsible for kicking off he distributed reduce
    // if we are we need to check if there has already been some singaling...
    else if(_rs->_rank == root_node) {

      // check if the node has signaled that it finished with the local reduces
      // if it did not mark that we are waiting for it...
      const auto it = std::find_if(reduce.done_nodes.begin(), reduce.done_nodes.end(), 
      [&](const command_t::tid_node_id_t &val) {
        return val.tid == _in.tid && val.node == _in.node; });

      const auto jt = std::find(reduce.waiting_for_nodes.begin(), reduce.waiting_for_nodes.end(), _in.node);

      if(it == reduce.done_nodes.end() && jt == reduce.waiting_for_nodes.end()) {
        reduce.waiting_for_nodes.push_back(_in.node);
      }
    }
  }

  // is this an purely local
  reduce.is_local = reduce.waiting_for_nodes.empty() && reduce.done_nodes.empty() && root_node == _rs->_rank;

  // copy the command
  reduce.command = std::move(_command);

  // if there are at least two inputs that are available on this node we can run stuff...
  _update_reduce(reduce);

  // we have more local command that we need to retire
  _left_reduce_to_retire++;

  return true;
}

void bbts::command_handler_reduce_t::_update_reduce(internal_reduce_state_t &reduce) {

  auto root_node = reduce.command->get_root_node();
  if(reduce.available_inputs.size() >= 2) {

    while (reduce.available_inputs.size() >= 2) {
      
      // find the out tid
      auto out_tid = reduce.missing_inputs.empty() && reduce.is_local ? reduce.command->get_outputs()[0].tid : -1;
      if(out_tid == -1) {
        _rs->_tensors_left_to_delete++;
      }

      // create the reduce
      auto Idx = reduce.available_inputs.size() - 1;
      auto partial_reduce = command_t::create_parital_reduce(reduce.command, {reduce.available_inputs[Idx - 1], reduce.available_inputs[Idx]}, out_tid, _rs->_rank);
      _rs->_heuristic->queue(std::move(partial_reduce));

      // remove the inputs
      reduce.available_inputs.pop_back();
      reduce.available_inputs.pop_back();

      // mark that one of the inpts is missing...
      reduce.missing_inputs.push_back(-1);
    }
  }
  // else if there is just one input and none of them are missing we need to notify that we are done
  // or if this is the root node we need to kick of a distributed reduce...
  else if(reduce.missing_inputs.empty() && reduce.available_inputs.size() == 1) {

    if(_rs->_rank == root_node && reduce.waiting_for_nodes.empty()) {
      
      auto partial_reduce = command_t::create_distributed_reduce(reduce.command, 
                                                                 {reduce.available_inputs[0], root_node}, 
                                                                 reduce.done_nodes);
      _rs->_heuristic->queue(std::move(partial_reduce));
    }
    else {

      // 
      _rs->_notify_done_reduces[root_node].enqueue_copy({reduce.command->id, reduce.available_inputs[0]});
    }
  }
}

void bbts::command_handler_reduce_t::tensor_available(command_id_t command_id, tid_t tid) {
  
  // since it is not an apply or move it must be a reduce
  auto reduce = reduce_commands.find(command_id);
  assert(reduce != reduce_commands.end());
  reduce->second.available_inputs.push_back(tid);

  // remove it from the missing inputs
  auto kt = std::find(reduce->second.missing_inputs.begin(), reduce->second.missing_inputs.end(), tid);
  *kt = *reduce->second.missing_inputs.rbegin();
  reduce->second.missing_inputs.pop_back();

  // update the reduce
  _update_reduce(reduce->second);
}

void bbts::command_handler_reduce_t::commands_finished_on_node(const std::vector<command_t::command_tid_id_t> &commands, node_id_t node) {

  // go through all the commands
  for(auto &ct : commands) {

    // go through tensors
    auto it = reduce_commands.find(ct.id);
    if(it == reduce_commands.end()) { continue; }

    // remove the node as we are not waiting for it
    const auto jt = std::find(it->second.waiting_for_nodes.begin(), it->second.waiting_for_nodes.end(), node);
    assert(jt != it->second.waiting_for_nodes.end());
    *jt = *it->second.waiting_for_nodes.rbegin();
    it->second.waiting_for_nodes.pop_back();

    // add it to the done nodes
    it->second.done_nodes.push_back(command_t::tid_node_id_t{.tid = ct.tid, .node = node });

    // update the reduce
    _update_reduce(it->second);
  }
}

bool bbts::command_handler_reduce_t::is_done() {
  return _left_reduce_to_retire == 0;
}

void bbts::command_handler_reduce_t::clear() {
  reduce_commands.clear();
  _left_reduce_to_retire = 0;
}

}