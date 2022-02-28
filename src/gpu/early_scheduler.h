#pragma once

#include "types.h"

namespace bbts {

class early_scheduler_t {
public:
  void tensor_loaded(tid_t id, int dev){};

  void tensor_unloaded(tid_t id, int dev){};

  void register_apply(const bbts::command_ptr_t &cmd){};

  void register_reduce(const bbts::command_ptr_t &cmd){};

  int32_t has_same_gpu() { return -1; };

  bool has_on_gpu() { return false; };

  kernel_prep_ptr_t get_next() { return nullptr; };
};

} // namespace bbts