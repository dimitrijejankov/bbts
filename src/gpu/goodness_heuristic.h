#pragma once

#include "types.h"

namespace bbts {

class goodness_heuristic_class_t {
public:
  void tensor_loaded(tid_t id) {};

  void tensor_unloaded(tid_t id) {};

  void register_apply(const bbts::command_ptr_t &cmd) {};

  void register_reduce(const bbts::command_ptr_t &cmd) {};

  kernel_prep_ptr_t get_next() { return nullptr; };
};

}