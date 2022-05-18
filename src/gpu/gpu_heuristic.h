#pragma once

#include "types.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <vector>

namespace bbts {


// the heuristic is GOODNESS = (num of inputs to copy, how many other commands need the same inputs)
class gpu_heuristic_t {
public:

  // make a heuristic with multiple device
  gpu_heuristic_t(uint32_t num_devices);

  // returns a kernel that does not need any copies and the GPU to run it on
  std::tuple<kernel_prep_ptr_t, int32_t>
  get_next_on_same(int32_t preffered_dev);

  // returns a kernel that needs just GPU copies
  kernel_prep_ptr_t get_next_on_any();

  // return a kernel that need CPU copies
  kernel_prep_ptr_t get_next_heuristic();

  // signal that a tensor is loaded on a GPU
  void tensor_loaded(tid_t id, int dev);

  // signal that a tensor is unloaded on a GPU
  void tensor_unloaded(tid_t id, int dev);

  // signal that a tensor is available in the CPU memory
  void tensor_on_cpu(tid_t id);

  // remove a tensor from the scheduler
  void remove_tensor(tid_t id);

  // register a new APPLY command to be scheduled
  void register_apply(bbts::gpu_command_schedule_ptr_t &apply_sch);
  
  // register a new REDUCE command to be scheduled
  void register_reduce(bbts::gpu_command_schedule_ptr_t &reduce_sch);

  // mark that a kernel has been scheduled
  void mark_as_scheduled(const kernel_prep_ptr_t &prep);

  // check if the heuristic has anything to schedule
  bool has_something();

private:

  struct heuristic_map_cmp_t {
      bool operator()(const std::tuple<int32_t, int32_t> &lhs, 
                      const std::tuple<int32_t, int32_t> &rhs) const {

          if (std::get<0>(lhs) == std::get<0>(rhs)) {
            return std::get<1>(lhs) < std::get<1>(rhs);
          }
          return std::get<0>(lhs) > std::get<0>(rhs);
      }
  };

  using heuristic_map_t = std::multimap<std::tuple<int32_t, int32_t>, std::tuple<command_id_t, command_t::op_type_t>, heuristic_map_cmp_t>;

  struct es_apply_command_nfo_t {

    // the id of the command
    command_id_t id;

    // how many inputs are there
    uint32_t num_inputs;

    // how many outputs are there
    uint32_t num_outputs;

    // how many tensors are actually available
    uint32_t inputs_available;

    // number of inputs that are loaded onto at least one GPU
    uint32_t loaded_inputs;
    
    // for each device we keep a cound on how many input tensors are loaded on that particular device
    std::array<u_int32_t, BBTS_MAX_GPU_DEVICES> inputs_on_devices = {0};
    
    // the tensor ids of the input to the kernel
    std::vector<tid_t> input_tids;

    // the size of each input tensor
    std::vector<size_t> input_sizes;

    // the tensor ids of the output of the kernel
    std::vector<tid_t> output_tids;

    // the size of each output tensor
    std::vector<size_t> output_sizes;

    // the kernel
    kernel_run_ptr_t run_me;
    
    // the iterator to the heuristic
    heuristic_map_t::iterator it;
  };

  struct es_reduce_command_nfo_t {

    // the id of the command
    command_id_t id;

    // how many inputs does the reduce have
    uint32_t num_inputs;

    // how many times have we issued a partial reduce (should be a total num_inputs - 1 at the end)
    uint32_t num_issued;

    // the output tid of thefinal result
    tid_t output_tid;

    // list of inputs exclusively on the CPU, that means NOT on GPU
    std::vector<tid_t> cpu_inputs;

    // we store here a list of all the inputs currently available on all the GPUs
    std::vector<tid_t> gpu_inputs;

    // TODO: this is not actually corrent and should be fixed in the future
    size_t output_size;

    // here we store a list of all the inputs available per GPU
    std::array<std::vector<tid_t>, BBTS_MAX_GPU_DEVICES> inputs_on_devices;

    // run this kernel
    kernel_run_ptr_t run_me;

    // the iterator to the heuristic
    heuristic_map_t::iterator it;
  };

  struct es_tensor_nfo {

    // should delete
    bool should_delete = false;

    // is this tensor on the CPU
    bool on_cpu = false;

    // how many GPU copies of the tensor do we have
    int32_t gpu_copies = 0;

    // it this on the device
    std::array<bool, BBTS_MAX_GPU_DEVICES> on_device;
  };

  uint32_t _calculate_heuristic_apply(const std::vector<tid_t> inputs);

  uint32_t _calculate_heuristic_reduce(const std::vector<tid_t> inputs);

  void _update_heuristic_for_apply(command_id_t id);

  void _update_heuristic_for_reduce(command_id_t id);

  void _update_heuristic_for_inputs(const std::vector<tid_t> &inputs);

  kernel_prep_ptr_t _create_reduce(command_id_t cmd);

  kernel_prep_ptr_t _create_apply(command_id_t cmd);

  void _unlink_command_from_tensor(tid_t id, command_id_t cmd);

  // how many gpus do we actually have
  uint32_t num_devices = 1;

  // (number of inputs not on GPU, where less is better, number of tensors used by other commands more is better)
  heuristic_map_t goodness_heuristic;

  std::unordered_multimap<tid_t, std::tuple<command_id_t, command_t::op_type_t>> tensors_to_cmds;

  // manages the information about the apply commands
  std::unordered_map<command_id_t, es_apply_command_nfo_t> apply_cmds;
  std::unordered_set<command_id_t> apply_in_gpu_memory;
  std::vector<std::unordered_set<command_id_t>> on_apply_single_gpu;
  
  // manages the information about the reduce commands 
  std::unordered_map<command_id_t, es_reduce_command_nfo_t> reduce_cmds;
  std::unordered_set<command_id_t> reduce_in_gpu_memory;
  std::vector<std::unordered_set<command_id_t>> on_reduce_single_gpu;
  
  // keeps track of all the tensors on the GPU (we need that)
  std::unordered_map<tid_t, es_tensor_nfo> tensors;

  // we assign these for anonymous 
  tid_t inner_anon_id = 0;

  // we assign the an id to each kernel prep
  int32_t cur_prep_id = 0;
};

} // namespace bbts