#pragma once

#include "../ud_functions/ud_function.h"
#include "../ud_functions/udf_manager.h"
#include "abstract_command.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace bbts {

class cost_model_t {
private:
  // the matched function
  struct function_match_t {

    // the cpu implementation
    ud_impl_t *cpu;

    // the gpu implementation
    ud_impl_t *gpu;
  };

public:
  struct transfer_cost_t {

    // the cost to transfer this tensor to the GPU
    float gpu_transfer_cost;

    // the cost to transfer this tensor to the CPU
    float network_transfer_cost;
  };

  struct kernel_cost_t {

    // the cost to run this kernel on the gpu
    float gpu = -1;

    // the cost to run this kernel on the cpu
    float cpu = -1;

    // check if this kernel is runnining on the gpu
    bool is_gpu() const { return gpu >= 0.0f; }

    // check if this kernel is running on the cpu
    bool is_cpu() const { return cpu >= 0.0f; }
  };

  // the choice we made
  struct ud_choice_t {

    // the cost
    float cost;

    // is this using the gpu
    bool is_gpu;

    // the pointer
    ud_impl_t *ud;
  };

  cost_model_t(std::unordered_map<tid_t, tensor_meta_t> meta,
               const std::vector<abstract_ud_spec_t> &funs,
               tensor_factory_ptr_t tf, udf_manager_ptr manager,
               float gpu_transfer_cost_per_byte, float send_cost_per_byte)
      : meta(std::move(meta)), tf(std::move(tf)), udm(std::move(manager)),
        send_cost_per_byte(send_cost_per_byte),
        gpu_transfer_cost_per_byte(gpu_transfer_cost_per_byte) {

    for (const auto &f : funs) {

      // get the matcher
      auto matcher = udm->get_matcher_for(f.ud_name);
      if (matcher == nullptr) {
        throw std::runtime_error("Could not match ud " + f.ud_name);
      }

      // the inplementation
      auto impl_cpu = matcher->findMatch(f.input_types, f.output_types, false);
      auto impl_gpu = matcher->findMatch(f.input_types, f.output_types, true);

      if (impl_cpu == nullptr && impl_gpu == nullptr) {
        throw std::runtime_error("Could not find a matching ud function.");
      }

      // found the match store it
      matched_functions[f.id] =
          function_match_t{.cpu = impl_cpu, .gpu = impl_gpu};
    }
  }

  // precompute the cost of all the ud executions
  void precompute_costs(const std::vector<abstract_command_t> &cmds) {

    ud_impl_t::tensor_params_t _params;
    bbts::ud_impl_t::meta_args_t input_meta;
    bbts::ud_impl_t::meta_args_t output_meta;

    // go through all the commands
    kernel_costs.resize(cmds.size());
    for (auto idx = 0; idx < cmds.size(); ++idx) {

      // get the command and the functions
      auto &cmd = cmds[idx];
      if (cmd.type == abstract_command_type_t::DELETE) {
        continue;
      }

      auto &ud = matched_functions[cmd.ud_id];

      // prepare the inputs
      input_meta.resize(cmd.input_tids.size());
      for (int i = 0; i < cmd.input_tids.size(); i++) {
        input_meta.set(i, meta[cmd.input_tids[i]]);
      }

      // prepare the outputs
      output_meta.resize(cmd.output_tids.size());
      for (int i = 0; i < cmd.output_tids.size(); i++) {
        output_meta.set(i, meta[cmd.output_tids[i]]);
      }

      // init the parameters
      _params = ud_impl_t::tensor_params_t{
          ._params = command_param_list_t{._data = cmd.params.data(),
                                          ._num_elements = cmd.params.size()}};

      // check if there is any kernel
      if (ud.cpu == nullptr && ud.gpu == nullptr) {
        throw std::runtime_error("Could not find an appropriate kernel." +
                                 std::string("The UD function is ") +
                                 std::to_string(cmd.ud_id) + "\n");
      }

      // check if there is an CPU kernel
      auto &ker_cost = kernel_costs[idx];
      if (ud.cpu != nullptr) {

        // get the output meta
        ud.cpu->get_out_meta(_params, input_meta, output_meta);

        // get the cost of running the kernel
        ker_cost.cpu = ud.cpu->get_complexity_hint(_params, input_meta);

        // store the meta
        for (auto i = 0; i < cmd.output_tids.size(); i++) {
          meta[cmd.output_tids[i]] = output_meta.get_by_idx(i);
        }
      }
      else if (ud.gpu != nullptr) {

        // get the output meta for the gpu
        ud.gpu->get_out_meta(_params, input_meta, output_meta);

        // get the cost of running the kernel
        ker_cost.gpu = ud.gpu->get_complexity_hint(_params, input_meta);

        // store the meta
        for (auto i = 0; i < cmd.output_tids.size(); i++) {
          meta[cmd.output_tids[i]] = output_meta.get_by_idx(i);
        }
      }
    }
  }

  transfer_cost_t get_transfer_cost(tid_t id) const {

    // try to find the meta for the tensor
    auto it = meta.find(id);
    if (it == meta.end()) {
      throw std::runtime_error("Could not find the tensor with tid " +
                               std::to_string(id) + "\n");
    }

    // calculate the costs
    auto size = tf->get_tensor_size(it->second);
    return transfer_cost_t{.gpu_transfer_cost =
                               size * gpu_transfer_cost_per_byte,
                           .network_transfer_cost = size * send_cost_per_byte};
  }

  function_match_t get_ud_choice(abstract_ud_spec_id_t id) {
    auto it = matched_functions.find(id);
    if (it == matched_functions.end()) {
      throw std::runtime_error("Could not find matched function.");
    }
    return it->second;
  }

  kernel_cost_t get_execution_cost(uint32_t cmd_id) const {
    if (cmd_id >= kernel_costs.size()) {
      throw std::runtime_error("Could not find the id.");
    }
    return kernel_costs[cmd_id];
  }

private:
  // all the matched functions
  std::unordered_map<abstract_ud_spec_id_t, function_match_t> matched_functions;

  // the meta so far in the cost model
  std::unordered_map<tid_t, tensor_meta_t> meta;

  // the kernel costs
  std::vector<kernel_cost_t> kernel_costs;

  // the tensor factory
  tensor_factory_ptr_t tf;

  // the udf manager
  udf_manager_ptr udm;

  // what is the extimated cost of moving bytes
  float send_cost_per_byte;

  // what is the estimated cost of moving bytes to the gpu
  float gpu_transfer_cost_per_byte;

  // we use this so that we don't keep allocating
  ud_impl_t::meta_args_t input_meta;
  std::vector<tensor_meta_t> _input_meta_vec;

  // the output meta so that we don't keep allocating
  ud_impl_t::meta_args_t out_meta;
  std::vector<tensor_meta_t> _output_meta_vec;
};

using cost_model_ptr_t = std::shared_ptr<cost_model_t>;

} // namespace bbts