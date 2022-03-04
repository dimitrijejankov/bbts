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

  gpu_heuristic_t(uint32_t num_devices) : num_devices(num_devices) {}

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

    // here we store a list of all the inputs available per GPU
    std::array<std::vector<tid_t>, BBTS_MAX_GPU_DEVICES> inputs_on_devices;
  };

  struct es_tensor_nfo {

    int32_t copies = 0;

    std::array<bool, BBTS_MAX_GPU_DEVICES> on_device;
  };

  void tensor_loaded(tid_t id, int dev) {
    
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
          apply_in_gpu_memory.insert(command_t::APPLY);
        }

        // can we schedule it on a single device
        if(apply_cmds[command_id].inputs_on_devices[dev] == apply_cmds[command_id].num_inputs) {
          on_apply_single_gpu[dev].insert(command_t::APPLY);
        }
      }
      else if(command_type == command_t::REDUCE) {
        
        // mark that we have it now on the device
        auto &cmd = reduce_cmds[command_id];
        cmd.inputs_on_devices[dev].push_back(id);

        // add this input if we don't already have it
        auto it = std::find(cmd.inputs.begin(), cmd.inputs.end(), id);
        if(it == cmd.inputs.end()) {
          cmd.inputs.push_back(id);
        }

        // if there are at least two inputs of a reduce it can be scheduled
        if(cmd.inputs.size() == 2) {
          reduce_in_gpu_memory.insert(id);
        }

        // same is true when we look at the per GPU case
        if(cmd.inputs_on_devices[dev].size() == 2) {
          on_reduce_single_gpu[dev].insert(id);
        }
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
          apply_in_gpu_memory.erase(command_id);
        }

        // should we unschedule it from a particular device
        if(apply_cmds[command_id].inputs_on_devices[dev] != apply_cmds[command_id].num_inputs) {
          on_apply_single_gpu[dev].erase(command_t::APPLY);
        }
      }
      else if(command_type == command_t::REDUCE) {

        // get the command
        auto &cmd = reduce_cmds[command_id];
          
        // if this is the last copy of the tensor 
        if(t.copies == 0) {
          
          // find it the tensor (not finding it is a bug)
          auto it = std::find(cmd.inputs.begin(), cmd.inputs.end(), id);
          assert(it != cmd.inputs.end());

          // remove it
          std::swap(*it, *(cmd.inputs.end() - 1));
          cmd.inputs.pop_back();

          // if we dropped to one input we need to unschedule it
          if(cmd.inputs.size() == 1) {
            reduce_in_gpu_memory.erase(command_id);
          }
        }

        // find the tensor and remove it, not finding it is a bug
        auto &dev_inputs = cmd.inputs_on_devices[dev];
        auto it = std::find(dev_inputs.begin(), dev_inputs.end(), id);
        std::swap(*it, *(dev_inputs.end() - 1));
        dev_inputs.pop_back();

        if(dev_inputs.size() == 1) {
          on_reduce_single_gpu[dev].erase(command_id);
        }
      }
    }
  };

  void tensor_on_cpu(tid_t id) {

  }

  void register_apply(const bbts::command_ptr_t &cmd){

    auto &apply_cmd = apply_cmds[cmd->id];

    apply_cmd.id = cmd->id;
    apply_cmd.num_inputs = cmd->get_num_inputs();
    apply_cmd.loaded_inputs = 0;

    for(int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

      auto in_tid = cmd->get_inputs()[idx].tid;
      auto &in = tensors[in_tid];
      
      // do we have it on any of the GPUs
      if(in.copies > 0) {
        apply_cmd.loaded_inputs++;
      }

      // update the per device count
      for(int32_t dev = 0; dev < num_devices; ++dev) {
        if(in.on_device[dev]) {
          apply_cmd.inputs_on_devices[dev]++;
        }
      }

      // add a link
      tensors_to_cmds.insert({in_tid, {apply_cmd.id, command_t::APPLY}});
    }

    // check if we have enough inputs on any GPU
    if(apply_cmd.loaded_inputs == apply_cmd.num_inputs) {
      apply_in_gpu_memory.insert(apply_cmd.id);
    }

    // check if we have enough inputs per GPU
    for(int32_t dev = 0; dev < num_devices; ++dev) {
      if(apply_cmd.inputs_on_devices[dev] == num_devices) {
        on_apply_single_gpu[dev].insert(apply_cmd.id);
      }
    }
  };

  void register_reduce(const bbts::command_ptr_t &cmd){

    auto &reduce_cmd = reduce_cmds[cmd->id];

    reduce_cmd.id = cmd->id;
    reduce_cmd.num_inputs = cmd->get_num_inputs();
    reduce_cmd.num_issued = 0;

    for(int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

      auto in_tid = cmd->get_inputs()[idx].tid;
      auto &in = tensors[in_tid];
      
      // do we have it on any of the GPUs
      if(in.copies > 0) {
        reduce_cmd.inputs.push_back(in_tid);
      }

      // update the per device count
      for(int32_t dev = 0; dev < num_devices; ++dev) {
        if(in.on_device[dev]) {
          reduce_cmd.inputs_on_devices[dev].push_back(in_tid);
        }
      }

      // add a link
      tensors_to_cmds.insert({in_tid, {reduce_cmd.id, command_t::REDUCE}});
    }

    // check if we have enough inputs on any GPU
    if(reduce_cmd.inputs.size() >= 2) {
      reduce_in_gpu_memory.insert(reduce_cmd.id);
    }

    // check if we have enough inputs per GPU
    for(int32_t dev = 0; dev < num_devices; ++dev) {
      if(reduce_cmd.inputs_on_devices[dev].size() >= 2) {
        on_reduce_single_gpu[dev].insert(reduce_cmd.id);
      }
    }
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

  // how many gpus do we actually have
  uint32_t num_devices = 1;

  std::unordered_multimap<tid_t, std::tuple<command_id_t, command_t::op_type_t>> tensors_to_cmds;

  // manages the information about the apply commands
  std::unordered_map<command_id_t, es_apply_command_nfo_t> apply_cmds;
  std::unordered_set<command_id_t> apply_in_gpu_memory;
  std::array<std::unordered_set<command_id_t>, BBTS_MAX_GPU_DEVICES> on_apply_single_gpu;
  
  // manages the information about the reduce commands 
  std::unordered_map<command_id_t, es_reduce_command_nfo_t> reduce_cmds;
  std::unordered_set<command_id_t> reduce_in_gpu_memory;
  std::array<std::unordered_set<command_id_t>, BBTS_MAX_GPU_DEVICES> on_reduce_single_gpu;
  
  // keeps track of all the tensors on the GPU (we need that)
  std::unordered_map<tid_t, es_tensor_nfo> tensors;

  // we assign these for anonymous 
  tid_t inner_anon_id = -1;
};

} // namespace bbts