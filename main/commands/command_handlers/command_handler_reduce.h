#pragma once

#include "command_handler.h"

namespace bbts {

class command_handler_reduce_t : public command_handler_t {
public:
  command_handler_reduce_t(class reservation_station_t *_rs);
  ~command_handler_reduce_t() override;

  bool retire_command(command_ptr_t _command) override;
  bool schedule_command(command_ptr_t _command) override;
  void tensor_available(command_id_t command_id, tid_t tid) override;
  void commands_finished_on_node(
      const std::vector<command_t::command_tid_id_t> &commands,
      node_id_t node) override;
  bool is_done() override;
  void clear() override;

  struct internal_reduce_state_t {

    // the inputs that need to be created here that the reduce is waiting for...
    std::vector<tid_t> missing_inputs;

    // the currently available inputs
    std::vector<tid_t> available_inputs;

    // we keep track of what nodes have finished all the local reduces they
    // could they notify this node once they are done, this is kept for just the
    // node that initiates the reduce
    std::vector<node_id_t> waiting_for_nodes;
    std::vector<command_t::tid_node_id_t> done_nodes;

    command_ptr_t command;

    bool is_local = false;
    };
    void _update_reduce(internal_reduce_state_t &reduce);

    // reduce commands we are keeping track of
    std::unordered_map<command_id_t, internal_reduce_state_t> reduce_commands;

    // how many moves are left to retire
    size_t _left_reduce_to_retire = 0;
};

}