#pragma once

#include "types.h"
#include <array>
#include <cassert>
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


// the heuristic is GOODNESS = sum number of commands using the same input
class gpu_heuristic_t {
public:

  struct heuristic_map_cmp_t {
      bool operator()(const std::tuple<int32_t, int32_t> &lhs, 
                      const std::tuple<int32_t, int32_t> &rhs) const {

          if (std::get<0>(lhs) == std::get<0>(rhs)) {
            return std::get<1>(lhs) > std::get<1>(rhs);
          }
          return std::get<0>(lhs) < std::get<0>(rhs);
      }
  };

  using heuristic_map_t = std::multimap<std::tuple<int32_t, int32_t>, std::tuple<command_id_t, command_t::op_type_t>, heuristic_map_cmp_t>;

  gpu_heuristic_t(uint32_t num_devices);

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

    // the tensor ids of the output of the kernel
    std::vector<tid_t> output_tids;

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

    // list of inputs exclusively on the CPU
    std::vector<std::tuple<tid_t, uint32_t>> cpu_inputs;

    // we store here a list of all the inputs currently available on all the GPUs
    std::vector<tid_t> loaded_inputs;

    // here we store a list of all the inputs available per GPU
    std::array<std::vector<tid_t>, BBTS_MAX_GPU_DEVICES> inputs_on_devices;

    // run this kernel
    kernel_run_ptr_t run_me;

    // the iterator to the heuristic
    heuristic_map_t::iterator it;
  };

  struct es_tensor_nfo {

    // is this tensor on the CPU
    bool on_cpu = false;

    // how many GPU copies of the tensor do we have
    int32_t gpu_copies = 0;

    // it this on the device
    std::array<bool, BBTS_MAX_GPU_DEVICES> on_device;
  };

  void tensor_loaded(tid_t id, int dev);

  void tensor_unloaded(tid_t id, int dev);

  uint64_t calculate_heuristic(const std::vector<tid_t> inputs);

  void update_heuristic_for_apply(command_id_t id);

  void update_heuristic_for_reduce(command_id_t id);

  void tensor_update_heuristic(tid_t id, bool is_loaded);

  void tensor_available_update_heuristic(tid_t id);

  void tensor_on_cpu(tid_t id);

  void register_apply(bbts::apply_schedule_ptr_t &apply_sch);

  void register_reduce(bbts::reduce_schedule_ptr_t &reduce_sch);

  kernel_prep_ptr_t create_reduce(command_id_t cmd);

  kernel_prep_ptr_t create_apply(command_id_t cmd);

  // returns the commands and the device to schedule it on
  std::tuple<kernel_prep_ptr_t, int32_t>
  get_next_on_same(int32_t preffered_dev);

  // returns the commands and the device to schedule it on
  kernel_prep_ptr_t get_next_on_any();

  kernel_prep_ptr_t get_next_heuristic();

  void mark_as_scheduled(const kernel_prep_ptr_t &prep);

  void remove_tensor(tid_t id);

  void _unlink_command_from_tensor(tid_t id, command_id_t cmd);

  // how many gpus do we actually have
  uint32_t num_devices = 1;

  // (number of inputs not on GPU, where less is better, number )
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
  tid_t inner_anon_id = -1;
};

} // namespace bbts