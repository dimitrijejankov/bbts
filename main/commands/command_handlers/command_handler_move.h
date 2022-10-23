#pragma once

#include "command_handler.h"

namespace bbts {

class command_handler_move_t : public command_handler_t {
public:

    command_handler_move_t(class reservation_station_t *_rs);
    ~command_handler_move_t() override;

    bool retire_command(command_ptr_t _command) override;
    bool schedule_command(command_ptr_t _command) override;
    void tensor_available(command_id_t command_id, tid_t tid) override;
    bool is_done() override;
    void clear() override;

    // the local apply and move commands and the number of tensors they are waiting for
    std::unordered_map<command_id_t, std::pair<command_ptr_t, int32_t>> _local_commands;

    // how many moves are left to retire
    size_t _left_local_to_retire = 0;
};

}