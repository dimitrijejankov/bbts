#include "command_handler_apply.h"
#include "../command.h"
#include "../reservation_station.h"

namespace bbts {

bool command_handler_apply_t::retire_command(command_ptr_t _command) {

  // make sure to go through the created tensors
  for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

    // get the tensor required in the output
    auto out = _command->get_output(i);

    // if this tensor is not on our node we don't do anything
    if(out.node != _rs->get_rank()) {
      continue;
    }

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

  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto in = _command->get_input(i);

    // if this tensor is not on our node we don't do anything
    if(in.node != _rs->get_rank()) {
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
  _local_commands.erase(_command->id);
 
  // we have one less to retire
  _left_local_to_retire--;
  _rs->_check_if_finished();

  return true;
}

bool command_handler_apply_t::schedule_command(command_ptr_t _command) {
    
  // count the number of inputs that are not present
  assert(_command->type == command_t::APPLY);
  int32_t num_not_present = 0;
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // all inputs must be local
    assert(_in.node == _rs->get_rank());

    // if it is a local we need to check if it was created already
    auto &s = _rs->_tensors[_in.tid];

    // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
    if(s.scheduled_for_delition) { return false; }

    // we are reading this tensor
    s.num_to_read++;

    // if it was not created we need to keep track of that
    if(!s.is_created) {

      // tensor is not present
      num_not_present++;

      // mark that this command is waiting
      _rs->_commands_waiting_for.insert({_in.tid, {_command->id, command_t::op_type_t::APPLY}});
    }
  }

  // go through the output tensors
  for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

    // grab the output tensor
    auto &_out = _command->get_output(idx);

    // check if the node
    if(_out.node == _rs->get_rank()) {

      // get the tid
      auto &s = _rs->_tensors[_out.tid];

      // make sure everything is fine TODO I need to recover from this somehow...
      if(s.scheduled_for_delition && s.is_created) { return false; }

      // we are writing to this tensor
      s.writing_tensor = true;
    }
  }

  // if we have all the required tensors we can kick off the command
  if(num_not_present == 0) {
    _rs->_reorder_buffer.queue(std::move(_command)); 
  }
  else {

    // store the number of tensors this command is waiting for
    auto cmd_id = _command->id;
    _local_commands[cmd_id] = { std::move(_command),  num_not_present };
  }

  // we have more local command that we need to retire
  _left_local_to_retire++;

  return true;
}

void bbts::command_handler_apply_t::tensor_available(command_id_t command_id, tid_t tid) {

  // try find a move or apply with that command id
  auto jt = _local_commands.find(command_id);
  if(jt != _local_commands.end()) {

    // check if we have all the inputs
    if(0 == (--jt->second.second)) {

      // schedule the command for execution
      _rs->_reorder_buffer.queue(std::move(jt->second.first));

      // remove the command
      _local_commands.erase(jt);
    }
  }
}

void bbts::command_handler_apply_t::commands_finished_on_node(const std::vector<command_t::command_tid_id_t> &commands, node_id_t node) {}

bool bbts::command_handler_apply_t::is_done() {
  return _left_local_to_retire == 0;
}

void bbts::command_handler_apply_t::clear() {
  _left_local_to_retire = 0;
  _local_commands.clear();
}

}