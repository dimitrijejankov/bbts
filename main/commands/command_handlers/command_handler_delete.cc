#include "command_handler_delete.h"
#include "../reservation_station.h"
#include <cassert>

namespace bbts {

command_handler_delete_t::command_handler_delete_t(reservation_station_t *_rs)
    : command_handler_t(_rs) {}

command_handler_delete_t::~command_handler_delete_t() {}

bool command_handler_delete_t::retire_command(command_ptr_t _command) {

  // delete commands are retired on a per tensor level 
  // so that logic is in the reservation station
  assert(false);
  return true;
}

bool command_handler_delete_t::schedule_command(command_ptr_t _command) {

  // go through the inputs and eiter remove the tensors directly,
  // or mark them for deletion if they are going to be used soon
  for(auto idx = 0; idx < _command->get_num_inputs(); ++idx) {

    // grab the input tensor
    auto &in = _command->get_input(idx);

    // mark the tensor as scheduled for deletion
    auto &s = _rs->_tensors[in.tid];

    // if we created the tensor, if not just delete it!
    if(s.is_created && s.num_to_read == 0 && !s.writing_tensor) {

      // remove the tensor immediately
      _rs->_remove_tensor(in.tid);
    }
    else {

      // ok the tensor is not ready for deletion schedule it
      assert(s.scheduled_for_delition == false);
      s.scheduled_for_delition = true;
    }

    // we got one more to delete
    _rs->_tensors_left_to_delete++;
  }

  // finish delete processed
  return true;
}

void bbts::command_handler_delete_t::tensor_available(command_id_t command_id, tid_t tid) {}

bool bbts::command_handler_delete_t::is_done() { 
  return _rs->_tensors_left_to_delete == 0;
}

void bbts::command_handler_delete_t::clear() { _rs->_tensors_left_to_delete = 0; };

}



