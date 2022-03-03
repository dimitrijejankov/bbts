#pragma once

#include "types.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <set>
#include <unordered_set>

namespace bbts {

class gpu_heuristic_t {
public:

  gpu_heuristic_t(uint32_t num_devices) : num_devices(num_devices) {
    
  }

  struct es_apply_command_nfo_t {

    command_id_t id;
    
    uint32_t num_inputs;

    uint32_t loaded_inputs;

    std::array<u_int32_t, BBTS_MAX_GPU_DEVICES> inputs_on_devices = {0};
  };

  struct es_reduce_command_nfo_t {

    command_id_t id;
    
    // how many inputs does the reduce have
    uint32_t num_inputs;

    // how many times have we issued a partial reduce (should be a total num_inputs - 1 at the end)
    uint32_t num_issued;

    // we store here a list of all the inputs currently available on all the GPUs
    std::vector<tid_t> inputs;
  };

  struct es_tensor_nfo {

    int32_t copies = 0;

    std::array<bool, BBTS_MAX_GPU_DEVICES> on_device;
  };

  void tensor_loaded(tid_t id, int dev){
    
    // mark that the tensor is now on the device
    auto &t = tensors[id];
    t.on_device[dev] = true;
    t.copies++;

    // find all the commands that use this tensor
    auto range = tensors_to_cmds.equal_range(id);
    
    // go through all commands and update
    for (auto it = range.first; it != range.second; ++it) {

      auto [command_id, command_type] = it->second;

      assert(command_type == command_t::APPLY || command_type == command_t::REDUCE);
      if(command_type == command_t::APPLY) {

        // update the counts
        apply_cmds[command_id].inputs_on_devices[dev]++;
        apply_cmds[command_id].loaded_inputs += t.copies == 1;
        
        // can we schedule it
        if(apply_cmds[command_id].loaded_inputs == apply_cmds[command_id].num_inputs) {
          apply_in_gpu_memory.insert({command_id, command_t::APPLY});
        }

        // can we schedule it on a single device
        if(apply_cmds[command_id].inputs_on_devices[dev] == apply_cmds[command_id].num_inputs) {
          on_apply_single_gpu[dev].insert({command_id, command_t::APPLY});
        }
      }
      else if(command_type == command_t::REDUCE){
        
      }
    }
  };

  void tensor_unloaded(tid_t id, int dev) {

    // mark that the tensor is now unloaded
    auto &t = tensors[id];
    t.on_device[dev] = false;
    t.copies--;

    // find all the commands that use this tensor
    auto range = tensors_to_cmds.equal_range(id);
    
    // go through all commands and update
    for (auto it = range.first; it != range.second; ++it) {

      auto [command_id, command_type] = it->second;
      assert(command_type == command_t::APPLY || command_type == command_t::REDUCE);

      if(command_type == command_t::APPLY) {
        apply_cmds[command_id].inputs_on_devices[dev]--;
        apply_cmds[command_id].loaded_inputs -= t.copies == 0;

        // should we unschedule it
        if(apply_cmds[command_id].loaded_inputs != apply_cmds[command_id].num_inputs) {
          apply_in_gpu_memory.erase({command_id, command_t::APPLY});
        }

        // should we unschedule it from a particular device
        if(apply_cmds[command_id].inputs_on_devices[dev] != apply_cmds[command_id].num_inputs) {
          on_apply_single_gpu[dev].erase({command_id, command_t::APPLY});
        }
      }
      else if(command_type == command_t::REDUCE) {
        
      }
    }
  };

  void register_apply(const bbts::command_ptr_t &cmd){

  };

  void register_reduce(const bbts::command_ptr_t &cmd){

  };

  // returns the commands and the device to schedule it on
  std::tuple<kernel_prep_ptr_t, int32_t> get_next_on_same(int32_t preffered_dev) {
    for(int32_t dev = 0; dev < num_devices; ++dev) {
      auto cur_dev = (preffered_dev + dev) % num_devices;
      if(!on_apply_single_gpu[cur_dev].empty()) {
        //*on_single_gpu[cur_dev].begin()
        return {nullptr, cur_dev};
      }
    }
    return {nullptr, -1}; 
  };
  
  // returns the commands and the device to schedule it on
  kernel_prep_ptr_t get_next_on_any() {
    if(!apply_in_gpu_memory.empty()) {
      // return *apply_in_gpu_memory.begin();
      return nullptr;
    }
    return nullptr;
  };

  kernel_prep_ptr_t get_next_heuristic() { 
    return nullptr; 
  };

  void mark_as_scheduled(const kernel_prep_ptr_t &prep) {
    
  }

  uint32_t num_devices = 1;

  std::unordered_map<command_id_t, es_apply_command_nfo_t> apply_cmds;

  std::unordered_multimap<tid_t, std::tuple<command_id_t, command_t::op_type_t>> tensors_to_cmds;

  std::set<std::tuple<command_id_t, command_t::op_type_t>> apply_in_gpu_memory;

  std::array<std::set<std::tuple<command_id_t, command_t::op_type_t>>, BBTS_MAX_GPU_DEVICES> on_apply_single_gpu;
  
  std::unordered_map<tid_t, es_tensor_nfo> tensors;

  tid_t inner_anon_id = -1;
};

} // namespace bbts