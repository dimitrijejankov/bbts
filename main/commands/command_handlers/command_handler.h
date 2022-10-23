#pragma once

#include "../command.h"

namespace bbts {

class command_handler_t {
public:

    command_handler_t(class reservation_station_t *_rs) : _rs(_rs) {}
    virtual ~command_handler_t() {};

    virtual bool retire_command(command_ptr_t _command) { return true; }

    virtual bool schedule_command(command_ptr_t _command) { return true; }

    virtual void tensor_available(command_id_t command_id, tid_t tid) {}

    virtual void commands_finished_on_node(const std::vector<command_t::command_tid_id_t> &commands, node_id_t node) {};

    virtual bool is_done() { return true; }

    virtual void clear() {}

protected:
    
    reservation_station_t *_rs;
};

}