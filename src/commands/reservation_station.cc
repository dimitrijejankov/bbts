#include "reservation_station.h"
#include "../server/static_config.h"
#include "command.h"
#include <cassert>
#include <utility>

bbts::reservation_station_t::reservation_station_t(bbts::node_id_t _node_id, int32_t num_nodes) : _rank(_node_id),
                                                                                                  _num_nodes(num_nodes),
                                                                                                  _notify_done_reduces(num_nodes) {}

bool bbts::reservation_station_t::queue_command(command_ptr_t _command) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  if(_command->type == command_t::APPLY) {
    return _queue_apply_command(std::move(_command));
  }
  else if(_command->type == command_t::REDUCE) {
    return _queue_reduce_command(std::move(_command));
  }
  else if(_command->type == command_t::MOVE) {
    return _queue_move_command(std::move(_command));
  }
  else {
    assert(_command->type == command_t::DELETE);
    return _queue_delete_command(std::move(_command));
  }
}

bool bbts::reservation_station_t::retire_command(command_ptr_t _command) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  // if this is a delete we remove the tensor
  if (_command->type == command_t::op_type_t::DELETE) {
    return _retire_remove(std::move(_command));
  }
  // handle the apply
  else if (_command->type == command_t::op_type_t::APPLY) {
    return _retire_apply(std::move(_command));
  }
  // handle the reduce
  else {
    return _retire_reduce(std::move(_command));
  }
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_move_command() {
  
  command_ptr_t out;
  if(!_heuristic.next_move(out)) {
    return nullptr;
  }
  return std::move(out);
}

bbts::command_ptr_t bbts::reservation_station_t::get_next_kernel_command() {
  command_ptr_t out;
  if(!_heuristic.next_apply(out)) {
    return nullptr;
  }
  return std::move(out);
}

bbts::command_ptr_t bbts::reservation_station_t::get_distributed_reduce_command() {
  command_ptr_t out;
  if(!_heuristic.next_reduce(out)) {
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
  auto cw = _commands_waiting_for[_rank].equal_range(_tid);
  for (auto it = cw.first; it != cw.second;) {

    // try to find the command
    auto jt = _local_commands.find(it->second);
    assert(jt != _local_commands.end());

    // check if we have all the inputs
    if(0 == (--jt->second.second)) {

      // schedule the command for execution
      _heuristic.queue_apply(std::move(jt->second.first));

      // remove the command
      _local_commands.erase(jt);
    }

    // remove the command from the waiting list
    it = _commands_waiting_for[_rank].erase(it);
  }
}

bbts::tid_t bbts::reservation_station_t::get_to_remove() {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  // wait until we have something to do
  // if the queue is shutdown just return -1
  tid_t t;
  bool success = _to_delete.wait_dequeue(t);
  return success ? t : -1;
}

void bbts::reservation_station_t::retire_remove(tid_t _tid) {

  // lock here
  std::unique_lock<std::mutex> lk(_m);

  // we got one less
  _left_to_delete--;
  if(_left_to_delete == 0 && _left_local_to_retire == 0 && _left_reduce_to_retire == 0) {
    _done_cv.notify_all();
  }
}

void bbts::reservation_station_t::shutdown() {

  // set the flag
  std::unique_lock<std::mutex> lk(_m);

  _shutdown = true;

  // notify that we are done
  _cv.notify_all();
  _to_delete.shutdown();
  for(auto &ndr : _notify_done_reduces) { ndr.shutdown(); }
}

void bbts::reservation_station_t::clear() {

  std::unique_lock<std::mutex> lk(_m);

  _last_cmd = -1;
  _heuristic.clear();
  _local_commands.clear();

  for(auto &ls : _commands_waiting_for) {
    ls.clear();
  }
  _tensors.clear();

  _heuristic.clear();
  for(auto &t : _notify_done_reduces) {
    t.clear();
  }

  _to_delete.clear();
}

void bbts::reservation_station_t::wait_until_finished() {

  // wait until all the commands are run
  std::unique_lock<std::mutex> lk(_m);
  _done_cv.wait(lk, [&]{
    return _left_local_to_retire == 0 && _left_reduce_to_retire == 0 && _left_to_delete == 0;
  });
}

void bbts::reservation_station_t::execute_scheduled_async() {

  // kick off everything
  std::unique_lock<std::mutex> lk(_m);
  _is_executing = true;
  _cv.notify_all();
}

void bbts::reservation_station_t::stop_executing() {

  // update the flag
  std::unique_lock<std::mutex> lk(_m);
  _is_executing = false;
  _last_cmd = -1;
}

void bbts::reservation_station_t::notify_ready_reduce(node_id_t node, const std::vector<command_id_t> &reduces) {

  // lock the tensor
  std::unique_lock<std::mutex> lk(_m);

  // go through tensors
}

[[nodiscard]] std::vector<bbts::tid_t> bbts::reservation_station_t::reduce_to_notify_node(node_id_t node, bool &is_done) {
  std::vector<bbts::tid_t> reduces;
  is_done = _notify_done_reduces[node].wait_dequeue_all(reduces);
  return std::move(reduces);
}

bool bbts::reservation_station_t::_retire_remove(command_ptr_t _command) {

    // remove the tensors
    for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

      // get the tensor required in the input
      auto t = _command->get_input(i);

      // remove the tensor immediately
      _remove_tensor(t.tid);

      // remove the command
      _local_commands.erase(_command->id);
    }

    return true;
}

bool bbts::reservation_station_t::_retire_apply(command_ptr_t _command) {

  // make sure to go through the created tensors
  for(int32_t i = 0; i < _command->get_num_outputs(); i++) {

    // get the tensor required in the output
    auto out = _command->get_output(i);

    // if this tensor is not on our node we don't do anything
    if(out.node != _rank) {
      continue;
    }

    // get the tid
    auto tid = out.tid;
    auto &s = _tensors[tid];

    // make sure that it was not created before
    assert(!s.is_created);

    // we are done writing to the tensor and
    s.is_created = true;
    s.writing_tensor = false;

    // remove the tensor if it is not needed
    if (s.num_to_read == 0 && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(out.tid);
    }

    // go through the commands that are waiting
    auto cw = _commands_waiting_for[_rank].equal_range(tid);
    for (auto it = cw.first; it != cw.second;) {

      // try to find the command
      auto jt = _local_commands.find(it->second);
      assert(jt != _local_commands.end());

      // check if we have all the inputs
      if (0 == (--jt->second.second)) {

        // schedule the command for execution
        _heuristic.queue_apply(std::move(jt->second.first));

        // remove the command
        _local_commands.erase(jt);
      }

      // remove the command from the waiting list
      it = _commands_waiting_for[_rank].erase(it);
    }


    ///TODO add the reduce stuff
  }

  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto in = _command->get_input(i);

    // if this tensor is not on our node we don't do anything
    if(in.node != _rank) {
      continue;
    }

    // get the tid
    auto tid = in.tid;
    auto &s = _tensors[tid];

    // decrement the number of readers
    s.num_to_read--;
    assert(s.num_to_read >= 0);

    // if there are no command that is writing to this tensor
    // reading this tensor and the tensor is scheduled for deletion, delete it here
    if (s.num_to_read == 0 && !s.writing_tensor && s.scheduled_for_delition) {

      // remove the tensor immediately
      _remove_tensor(in.tid);
    }
  }

  // remove the command
  _local_commands.erase(_command->id);
 
  // we have one less to retire
  _left_local_to_retire--;
  if(_left_to_delete == 0 && _left_local_to_retire == 0 && _left_reduce_to_retire == 0) {
    _done_cv.notify_all();
  }

  return true;
}

bool bbts::reservation_station_t::_retire_reduce(command_ptr_t _command) {

}

bool bbts::reservation_station_t::_queue_delete_command(command_ptr_t _command) {

  // go through the inputs and eiter remove the tensors directly,
  // or mark them for deletion if they are going to be used soon
  for(auto idx = 0; idx < _command->get_num_inputs(); ++idx) {

    // grab the input tensor
    auto &in = _command->get_input(idx);

    // mark the tensor as scheduled for deletion
    auto &s = _tensors[in.tid];

    // if we created the tensor, if not just delete it!
    if(s.is_created && s.num_to_read == 0 && !s.writing_tensor) {

      // remove the tensor immediately
      _remove_tensor(in.tid);
    }
    else {

      // ok the tensor is not ready for deletion schedule it
      assert(s.scheduled_for_delition == false);
      s.scheduled_for_delition = true;
    }

    // we got one more to delete
    _left_to_delete++;
  }

  // finish delete processed
  return true;
}

bool bbts::reservation_station_t::_queue_reduce_command(command_ptr_t _command) {

  auto &reduce = reduce_commands[_command->id];
  auto root_node = _command->get_root_node();

  // go through all the inputs
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // check if this is a remote tensor or a local tensor
    if(_in.node == _rank) {

      // if it is a local we need to check if it was created already
      auto &s = _tensors[_in.tid];

      // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
      if(s.scheduled_for_delition) { return false; }

      // we are reading this tensor
      s.num_to_read++;

      // if it was not created we need to keep track of that
      if(!s.is_created) {

        // mark that this command is waiting
        _commands_waiting_for[_rank].insert({_in.tid, _command->id});
        reduce.missing_inputs.push_back(_in.tid);
      }
      else {

        // the tensor is available so we add it here
        reduce.available_inputs.push_back(_in.tid);
      }
    }
    // if it is not a local tensor check if we are the node responsible for kicking off he distributed reduce
    // if we are we need to check if there has already been some singaling...
    else if(_rank == root_node) {

      // check if the node has signaled that it finished with the local reduces
      // if it did not mark that we are waiting for it...
      const auto it = std::find(reduce.done_nodes.begin(), reduce.done_nodes.end(), _in.tid);
      const auto jt = std::find(reduce.waiting_for_nodes.begin(), reduce.waiting_for_nodes.end(), _in.tid);
      if(it == reduce.done_nodes.end() && jt == reduce.waiting_for_nodes.end()) {
        reduce.waiting_for_nodes.push_back(_in.tid);
      }
    }
  }

  // if there are at least two inputs that are available on this node we can run stuff...
  if(reduce.available_inputs.size() >= 2) {

    // mark that we can run stuff here...
    // TODO most likely we need some heuristic update ehre
    _heuristic.queue_reduce(std::move(_command));
  }
  // else if there is just one input and none of them are missing we need to notify that we are done
  // or if this is the root node we need to kick of a distributed reduce...
  else if(reduce.missing_inputs.empty() && reduce.available_inputs.size() == 1) {

    if(_rank == root_node) {

      // TODO most likely we need some heuristic update ehre
      _heuristic.queue_reduce(std::move(_command));
    }
    else {
      _notify_done_reduces[root_node].enqueue(_command->id);
    }
  }

  // we have more local command that we need to retire
  _left_reduce_to_retire++;

  return true;
}

bool bbts::reservation_station_t::_queue_apply_command(command_ptr_t _command) {

  // count the number of inputs that are not present
  assert(_command->type == command_t::APPLY);
  int32_t num_not_present = 0;
  for(int32_t i = 0; i < _command->get_num_inputs(); i++) {

    // get the tensor required in the input
    auto _in = _command->get_input(i);

    // all inputs must be local
    assert(_in.node == _rank);

    // if it is a local we need to check if it was created already
    auto &s = _tensors[_in.tid];

    // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
    if(s.scheduled_for_delition) { return false; }

    // we are reading this tensor
    s.num_to_read++;

    // if it was not created we need to keep track of that
    if(!s.is_created) {

      // tensor is not present
      num_not_present++;

      // mark that this command is waiting
      _commands_waiting_for[_rank].insert({_in.tid, _command->id});
    }
  }

  // go through the output tensors
  for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

    // grab the output tensor
    auto &_out = _command->get_output(idx);

    // check if the node
    if(_out.node == _rank) {

      // get the tid
      auto &s = _tensors[_out.tid];

      // make sure everything is fine TODO I need to recover from this somehow...
      if(s.scheduled_for_delition && s.is_created) { return false; }

      // we are writing to this tensor
      s.writing_tensor = true;
    }
  }

  // if we have all the required tensors we can kick off the command
  if(num_not_present == 0) {
    _heuristic.queue_apply(std::move(_command)); 
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

bool bbts::reservation_station_t::_queue_move_command(command_ptr_t _command) {

  // get the tensor required in the input
  assert(_command->get_num_inputs() == 1);
  auto _in = _command->get_input(0);

  // check if this node is initiating the move or recieving from it
  if(_in.node == _rank) {

    // if it is a local we need to check if it was created already
    auto &s = _tensors[_in.tid];

    // make sure that this tensor was not deleted before this TODO I need to recover from this somehow...
    if(s.scheduled_for_delition) { return false; }

    // we are reading this tensor
    s.num_to_read++;

    // if it was not created we need to keep track of that
    if(!s.is_created) {

      // mark that this command is waiting
      _commands_waiting_for[_rank].insert({_in.tid, _command->id});

      // mark that 
      auto cmd_id = _command->id;
      _local_commands[cmd_id] = { std::move(_command),  1};
    }
    else {

      // schedule the move
      _heuristic.queue_move(std::move(_command));
    }

    // we have more local command that we need to retire
    _left_local_to_retire++;
  }
  else {
    
    // go through the output tensors
    for(auto idx = 0; idx < _command->get_num_outputs(); ++idx) {

      // grab the output tensor
      auto &_out = _command->get_output(idx);

      // check if the node
      if(_out.node == _rank) {

        // get the tid
        auto &s = _tensors[_out.tid];

        // make sure everything is fine TODO I need to recover from this somehow...
        assert(!(s.scheduled_for_delition && s.is_created));
        if(s.scheduled_for_delition && s.is_created) { return false; }

        // we are writing to this tensor
        s.writing_tensor = true;
      }
    }
  }

  // we are done here
  return true;
}

void bbts::reservation_station_t::_remove_tensor(tid_t tid) {

  // remove the tensor from the storage
  _to_delete.enqueue(tid);

  // remove the tensor
  _tensors.erase(tid);
}