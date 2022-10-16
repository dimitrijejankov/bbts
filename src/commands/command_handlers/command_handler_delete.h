#pragma once

#include "../command_handler.h"

namespace bbts {

class command_handler_delete_t : public command_handler_t {
public:

    command_handler_delete_t(class reservation_station_t *_rs) : command_handler_t(_rs) {}

    bool retire_command(command_ptr_t _command) override;
    bool schedule_command(command_ptr_t _command) override;
    void tensor_available(command_id_t command_id, tid_t tid) override;
    bool is_done() override;
    void clear() override;

};

}