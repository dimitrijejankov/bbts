#pragma once

#include "command_handler.h"
#include <cstddef>

namespace bbts {

class command_handler_apply_t : public command_handler_t {
public:

    command_handler_apply_t(class reservation_station_t *_rs) : command_handler_t(_rs) {}

    bool retire_command(command_ptr_t _command) override;
    bool schedule_command(command_ptr_t _command) override;
    void tensor_available(command_id_t command_id, tid_t tid) override;
    void commands_finished_on_node(const std::vector<command_t::command_tid_id_t> &commands, node_id_t node) override;
    bool is_done() override;
    void clear() override;
    
    // the local apply and move commands and the number of tensors they are waiting for
    std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

    size_t _left_local_to_retire = 0;
};

}