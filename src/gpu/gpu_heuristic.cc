#include "gpu_heuristic.h"
#include "types.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <tuple>

namespace bbts {

gpu_heuristic_t::gpu_heuristic_t(uint32_t num_devices)
    : num_devices(num_devices) {

  on_apply_single_gpu.resize(num_devices);
  on_reduce_single_gpu.resize(num_devices);

  apply_gpu_goodness_heuristic.resize(num_devices);
  reduce_gpu_goodness_heuristic.resize(num_devices);
}
void gpu_heuristic_t::tensor_loaded(tid_t id, int dev) {

  // mark that the tensor is now on the device
  auto &t = tensors[id];
  bool just_created = t.gpu_copies == 0 && !t.on_cpu;
  t.on_device[dev] = true;
  t.gpu_copies++;

  // find all the commands that use this tensor
  auto range = tensors_to_cmds.equal_range(id);

  // go through all commands and update
  for (auto it = range.first; it != range.second; ++it) {

    auto [command_id, command_type] = it->second;

    assert(command_type == command_t::APPLY ||
           command_type == command_t::REDUCE);
    if (command_type == command_t::APPLY) {

      // update the counts
      auto &apply = apply_cmds[command_id];
      apply.inputs_available += just_created;
      apply.inputs_on_devices[dev]++;
      apply.loaded_inputs += t.gpu_copies == 1;

      // can we schedule it
      if (apply.loaded_inputs == apply.num_inputs) {
        apply_in_gpu_memory.insert(command_id);
        _apply_in_gpu_memory_insert(command_id);
      }

      // can we schedule it on a single device
      if (apply.inputs_on_devices[dev] ==
          apply.num_inputs) {
        on_apply_single_gpu[dev].insert(command_id);
      }

      // update the heuristic for the apply
      _update_heuristic_for_apply(command_id);
      _update_gpu_heuristic_for_apply(command_id);

    } else if (command_type == command_t::REDUCE) {

      // mark that we have it now on the device
      auto &cmd = reduce_cmds[command_id];
      cmd.inputs_on_devices[dev].push_back(id);

      // mark that this tid is not now only exclusively on the CPU
      auto jt = std::find(cmd.cpu_inputs.begin(), cmd.cpu_inputs.end(), id);
      if (t.gpu_copies == 1 && jt != cmd.cpu_inputs.end()) {
        std::iter_swap(jt, (cmd.cpu_inputs.end() - 1));
        cmd.cpu_inputs.pop_back();
      }

      // add this input if we don't already have it
      auto it =
          std::find(cmd.gpu_inputs.begin(), cmd.gpu_inputs.end(), id);
      if (it == cmd.gpu_inputs.end()) {
        cmd.gpu_inputs.push_back(id);
      }

      // if there are at least two inputs of a reduce it can be scheduled
      if (cmd.gpu_inputs.size() == 2) {
        reduce_in_gpu_memory.insert(command_id);
        _reduce_in_gpu_memory_insert(command_id);
      }

      // same is true when we look at the per GPU case
      if (cmd.inputs_on_devices[dev].size() == 2) {
        on_reduce_single_gpu[dev].insert(command_id);
      }

      // update the heuristic for the reduce
      _update_heuristic_for_reduce(command_id);
      _update_gpu_heuristic_for_reduce(command_id);
    }
  }
};
void gpu_heuristic_t::tensor_unloaded(tid_t id, int dev) {

  // mark that the tensor is now unloaded
  auto &t = tensors[id];
  t.on_device[dev] = false;
  t.gpu_copies--;

  // find all the commands that use this tensor
  auto range = tensors_to_cmds.equal_range(id);

  // go through all commands and update the stats along with scheduling
  for (auto it = range.first; it != range.second; ++it) {

    auto [command_id, command_type] = it->second;
    assert(command_type == command_t::APPLY ||
           command_type == command_t::REDUCE);

    if (command_type == command_t::APPLY) {

      auto &apply = apply_cmds[command_id];

      apply.inputs_on_devices[dev]--;
      apply.loaded_inputs -= t.gpu_copies == 0;
      apply.inputs_available -= (t.gpu_copies == 0 && !t.on_cpu); 

      // should we unschedule it
      if (apply.loaded_inputs !=
          apply.num_inputs) {
        apply_in_gpu_memory.erase(command_id);
        _apply_in_gpu_memory_remove(command_id);
      }

      // should we unschedule it from a particular device
      if (apply.inputs_on_devices[dev] !=
          apply.num_inputs) {
        on_apply_single_gpu[dev].erase(command_id);
      }
    } else if (command_type == command_t::REDUCE) {

      // get the command
      auto &cmd = reduce_cmds[command_id];

      // if this is the last copy of the tensor
      if (t.gpu_copies == 0) {

        // find it the tensor (not finding it is a bug)
        auto it =
            std::find(cmd.gpu_inputs.begin(), cmd.gpu_inputs.end(), id);
        assert(it != cmd.gpu_inputs.end());

        // remove it
        std::swap(*it, *(cmd.gpu_inputs.end() - 1));
        cmd.gpu_inputs.pop_back();

        // if we dropped to one input we need to unschedule it
        if (cmd.gpu_inputs.size() == 1) {
          reduce_in_gpu_memory.erase(command_id);
          _reduce_in_gpu_memory_remove(command_id);
        }

        // if it is exclusively on the cpu set that
        if(t.on_cpu) {
          cmd.cpu_inputs.push_back(id);
        }

        // stuff changed update the heuristic
        _update_heuristic_for_reduce(command_id);
        _update_gpu_heuristic_for_reduce(command_id);
      }

      // find the tensor and remove it, not finding it is a bug
      auto &dev_inputs = cmd.inputs_on_devices[dev];
      auto it = std::find(dev_inputs.begin(), dev_inputs.end(), id);
      std::iter_swap(it, (dev_inputs.end() - 1));
      dev_inputs.pop_back();

      if (dev_inputs.size() == 1) {
        on_reduce_single_gpu[dev].erase(command_id);
      }
    }
  }

  // update the heuristic
  if (t.gpu_copies == 0) {
    // tensor_update_heuristic(id, false);
  }
};

uint32_t gpu_heuristic_t::_calculate_heuristic_apply(const std::vector<tid_t> inputs) {

  uint64_t total = 0;
  for (auto &in : inputs) {

    // do make sure we don't have it
    // as othewise we don't need to care about it
    if (tensors[in].gpu_copies != 0) {
      continue;
    }

    auto range = tensors_to_cmds.equal_range(in);
    total += std::distance(range.first, range.second);
  }
  return total;
}

uint32_t gpu_heuristic_t::_calculate_gpu_heuristic_apply(const std::vector<tid_t> inputs, int32_t dev) {

  uint64_t total = 0;
  for (auto &in : inputs) {

    // do make sure we don't have it
    // as othewise we don't need to care about it
    if (tensors[in].on_device[dev]) {
      continue;
    }

    auto range = tensors_to_cmds.equal_range(in);
    total += std::distance(range.first, range.second);
  }
  return total;
}

uint32_t gpu_heuristic_t::_calculate_heuristic_reduce(const std::vector<tid_t> inputs) {

  auto num_available = 0;
  uint32_t best = 0;
  uint32_t second_best = 0;
  for (auto &in : inputs) {

    num_available += tensors[in].gpu_copies != 0;
    if (tensors[in].gpu_copies != 0) {
      continue;
    }

    // if there are two inputs that are in GPU memory we will not laod anything if we run the the kernel
    if(num_available == 2) {
      return 0;
    }

    auto range = tensors_to_cmds.equal_range(in);
    auto value = std::max(best, (uint32_t) std::distance(range.first, range.second));

    if(best < value) {
      second_best = best;
      best = value;
    }
    else if(second_best < value) {
      second_best = value;
    }
  }

  return best + second_best;
}

uint32_t gpu_heuristic_t::_calculate_gpu_heuristic_reduce(const std::vector<tid_t> inputs, int32_t dev) {

  auto num_available = 0;
  uint32_t best = 0;
  uint32_t second_best = 0;
  for (auto &in : inputs) {

    num_available += tensors[in].on_device[dev];
    if (tensors[in].on_device[dev]) {
      continue;
    }

    // if there are two inputs that are in GPU memory of a single GPU
    // we will not laod anything if we run the the kernel as this is already enough
    if(num_available == 2) {
      return 0;
    }

    auto range = tensors_to_cmds.equal_range(in);
    auto value = std::max(best, (uint32_t) std::distance(range.first, range.second));

    if(best < value) {
      second_best = best;
      best = value;
    }
    else if(second_best < value) {
      second_best = value;
    }
  }

  return best + second_best;
}

void gpu_heuristic_t::_update_heuristic_for_apply(command_id_t id) {

  // remove it necessary
  auto &apply_cmd = apply_cmds[id];

  // check if we should even update it
  if(apply_cmd.inputs_available != apply_cmd.num_inputs) {
    return;
  }

  // remove the previous entry if necessary
  if(apply_cmd.it != goodness_heuristic.end()) {
    goodness_heuristic.erase(apply_cmd.it);
  }

  // form the goodness heuristic and insert it
  int32_t needed_inputs = apply_cmd.num_inputs - apply_cmd.loaded_inputs;
  int32_t heuristic_val = _calculate_heuristic_apply(apply_cmd.input_tids);
  apply_cmd.it = goodness_heuristic.insert({{needed_inputs, heuristic_val}, {id, command_t::APPLY}});
}

void gpu_heuristic_t::_update_heuristic_for_reduce(command_id_t id) {

  // remove it necessary
  auto &reduce_cmd = reduce_cmds[id];

  // remove the previous entry if necessary
  if(reduce_cmd.it != goodness_heuristic.end()) {
    goodness_heuristic.erase(reduce_cmd.it);
    reduce_cmd.it = goodness_heuristic.end();
  }

  // check if we should even update it
  if((reduce_cmd.cpu_inputs.size() + reduce_cmd.gpu_inputs.size()) < 2) {
    return;
  }

  // since the reduce kernel must be binary the only question is whether we already have an input on the GPU or not
  int32_t needed_inputs = reduce_cmd.gpu_inputs.empty() ? 2 : 1;
  int32_t heuristic_val = _calculate_heuristic_reduce(reduce_cmd.cpu_inputs);
  reduce_cmd.it = goodness_heuristic.insert({{needed_inputs, heuristic_val}, {id, command_t::REDUCE}});
}

void gpu_heuristic_t::_update_heuristic_for_inputs(const std::vector<tid_t> &inputs) {

  // update the heuristic for the commands that share inputs with it
  for (int32_t idx = 0; idx < inputs.size(); ++idx) {

    // get the input tid
    auto in_tid = inputs[idx];
    auto range = tensors_to_cmds.equal_range(in_tid);

    for(auto it = range.first; it != range.second; ++it) {

      // we just updated this command 
      auto [command, type] = it->second;

      // make sure it is the one of the two types of commands
      assert(type == command_t::APPLY || type == command_t::REDUCE);
      if(type == command_t::APPLY) {
        _update_heuristic_for_apply(command);
        _update_gpu_heuristic_for_apply(command);
      }
      else {
        _update_heuristic_for_reduce(command);
        _update_gpu_heuristic_for_reduce(command);
      }
    }
  }
}

void gpu_heuristic_t::register_apply(bbts::gpu_command_schedule_ptr_t &apply_sch) {

  auto &cmd = apply_sch->cmd;
  auto &apply_cmd = apply_cmds[cmd->id];

  apply_cmd.id = cmd->id;
  apply_cmd.num_inputs = cmd->get_num_inputs();
  apply_cmd.inputs_available = 0;
  apply_cmd.loaded_inputs = 0;
  apply_cmd.input_tids.reserve(cmd->get_num_inputs());
  apply_cmd.input_sizes = apply_sch->input_sizes;
  apply_cmd.output_tids.reserve(cmd->get_num_outputs());
  apply_cmd.output_sizes = apply_sch->output_sizes;
  apply_cmd.it = goodness_heuristic.end();
  for(auto dev = 0; dev < num_devices; ++dev) {
    apply_cmd.jts[dev] = apply_gpu_goodness_heuristic[dev].end();
  }
  
  apply_cmd.run_me = std::make_shared<kernel_run_t>();
  apply_cmd.run_me->ud = apply_sch->fn;
  apply_cmd.run_me->inputs.resize(cmd->get_num_inputs());
  apply_cmd.run_me->outputs.resize(cmd->get_num_outputs());
  apply_cmd.run_me->params = apply_sch->params;

  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    auto in_tid = cmd->get_inputs()[idx].tid;
    auto &in = tensors[in_tid];

    // do we have it on any of the GPUs
    if (in.gpu_copies > 0) {
      apply_cmd.loaded_inputs++;
    }

    // do we have the input at all if so mark it as available
    apply_cmd.inputs_available += in.gpu_copies || in.on_cpu;

    // update the per device count
    for (int32_t dev = 0; dev < num_devices; ++dev) {
      if (in.on_device[dev]) {
        apply_cmd.inputs_on_devices[dev]++;
      }
    }

    // add a link
    tensors_to_cmds.insert({in_tid, {apply_cmd.id, command_t::APPLY}});

    // store the input tid
    apply_cmd.input_tids.push_back(in_tid);
  }

  // are all the inputs available either on CPU or GPU if so we need to add it to the heuristic
  _update_heuristic_for_apply(apply_cmd.id);
  _update_gpu_heuristic_for_apply(apply_cmd.id);

  // next we need to update the heuristic for each other command with same inputs
  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    // get the input tid
    auto in_tid = cmd->get_inputs()[idx].tid;
    auto range = tensors_to_cmds.equal_range(in_tid);

    for(auto it = range.first; it != range.second; ++it) {

      // we just updated this command 
      auto [command, type] = it->second;
      if(apply_cmd.id == command) { continue; }

      // make sure it is the one of the two types of commands
      assert(type == command_t::APPLY || type == command_t::REDUCE);
      if(type == command_t::APPLY) {
        _update_heuristic_for_apply(command);
        _update_gpu_heuristic_for_apply(command);
      }
      else {
        _update_heuristic_for_reduce(command);
        _update_gpu_heuristic_for_reduce(command);
      }
    }
  }

  // store the output tids
  for (int32_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
    apply_cmd.output_tids.push_back(cmd->get_outputs()[idx].tid);
  }

  // check if we have enough inputs on any GPU
  if (apply_cmd.loaded_inputs == apply_cmd.num_inputs) {
    apply_in_gpu_memory.insert(apply_cmd.id);
    _apply_in_gpu_memory_insert(apply_cmd.id);
  }

  // check if we have enough inputs per GPU
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    if (apply_cmd.inputs_on_devices[dev] == apply_cmd.num_inputs) {
      on_apply_single_gpu[dev].insert(apply_cmd.id);
    }
  }
};

void gpu_heuristic_t::register_reduce(bbts::gpu_command_schedule_ptr_t &reduce_sch) {

  auto &cmd = reduce_sch->cmd;
  auto &reduce_cmd = reduce_cmds[cmd->id];

  reduce_cmd.id = cmd->id;
  reduce_cmd.num_inputs = cmd->get_num_inputs();
  reduce_cmd.num_issued = 0;

  // TODO: this is not corrent but it will for for testing
  reduce_cmd.output_size = reduce_sch->output_sizes.front();
  reduce_cmd.it = goodness_heuristic.end();
  for(auto dev = 0; dev < num_devices; ++dev) {
    reduce_cmd.jts[dev] = reduce_gpu_goodness_heuristic[dev].end();
  }

  // fill out the kernel
  reduce_cmd.run_me = std::make_shared<kernel_run_t>();
  reduce_cmd.run_me->ud = reduce_sch->fn;
  reduce_cmd.run_me->inputs.resize(cmd->get_num_inputs());
  reduce_cmd.run_me->outputs.resize(cmd->get_num_outputs());
  reduce_cmd.run_me->params = reduce_sch->params;

  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    auto in_tid = cmd->get_inputs()[idx].tid;
    auto &in = tensors[in_tid];

    // do we have it on any of the GPUs
    if (in.gpu_copies > 0) {
      reduce_cmd.gpu_inputs.push_back(in_tid);
    }

    // update the per device count
    for (int32_t dev = 0; dev < num_devices; ++dev) {
      if (in.on_device[dev]) {
        reduce_cmd.inputs_on_devices[dev].push_back(in_tid);
      }
    }

    // check if the input is on the CPU
    if(in.on_cpu && in.gpu_copies == 0) {
      reduce_cmd.cpu_inputs.push_back(in_tid);
    }

    // add a link
    tensors_to_cmds.insert({in_tid, {reduce_cmd.id, command_t::REDUCE}});
  }

  // are all the inputs available either on CPU or GPU if so we need to add it to the heuristic
  _update_heuristic_for_reduce(reduce_cmd.id);
  _update_gpu_heuristic_for_reduce(reduce_cmd.id);

  // next we need to update the heuristic for each other command with same inputs
  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    // get the input tid
    auto in_tid = cmd->get_inputs()[idx].tid;
    auto range = tensors_to_cmds.equal_range(in_tid);

    for(auto it = range.first; it != range.second; ++it) {

      // we just updated this command 
      auto [command, type] = it->second;
      if(reduce_cmd.id == command) { continue; }

      // make sure it is the one of the two types of commands
      assert(type == command_t::APPLY || type == command_t::REDUCE);
      if(type == command_t::APPLY) {
        _update_heuristic_for_apply(command);
        _update_gpu_heuristic_for_apply(command);
      }
      else {
        _update_heuristic_for_reduce(command);
        _update_gpu_heuristic_for_reduce(command);
      }
    }
  }

  // set the output
  reduce_cmd.output_tid = cmd->get_output(0).tid;

  // check if we have enough inputs on any GPU
  if (reduce_cmd.gpu_inputs.size() >= 2) {
    reduce_in_gpu_memory.insert(reduce_cmd.id);
    _reduce_in_gpu_memory_insert(reduce_cmd.id);
  }

  // check if we have enough inputs per GPU
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    if (reduce_cmd.inputs_on_devices[dev].size() >= 2) {
      on_reduce_single_gpu[dev].insert(reduce_cmd.id);
    }
  }
};
kernel_prep_ptr_t gpu_heuristic_t::_create_reduce(command_id_t cmd) {

  // grab the reduce
  auto &reduce_cmd = reduce_cmds[cmd];

  // fill it out
  auto ret = std::make_shared<kernel_prep_t>();
  ret->command_id = reduce_cmd.id;
  ret->type = command_t::REDUCE;
  ret->cpu_done = false;
  ret->gpu_done = false;
  
  // TODO this is not corrent but will be fixed in the future
  ret->input_sizes = {reduce_cmd.output_size,  reduce_cmd.output_size};
  ret->output_sizes = {reduce_cmd.output_size};
  
  ret->cpu_transfers = {};
  ret->gpu_transfers = {};

  // check where to get the inputs
  assert((reduce_cmd.gpu_inputs.size() + reduce_cmd.cpu_inputs.size()) >= 2);
  if(reduce_cmd.gpu_inputs.size() >= 2) {
    ret->input = {reduce_cmd.gpu_inputs[0], 
                  reduce_cmd.gpu_inputs[1]};
  }
  else if(reduce_cmd.gpu_inputs.size() == 1) {
    ret->input = {reduce_cmd.gpu_inputs[0], 
                  reduce_cmd.cpu_inputs[0]};
  }
  else {
    ret->input = {reduce_cmd.cpu_inputs[0], 
                  reduce_cmd.cpu_inputs[1]};
  }

  // check if this is the last time would be issuing a partial reduce
  // we need to issue it reduce_cmd.num_inputs - 1
  if (reduce_cmd.num_issued + 1 == reduce_cmd.num_inputs - 1) {
    ret->output = {reduce_cmd.output_tid};
  } else {
    // get the new anonymous tensor
    ret->output = {inner_anon_id - 1};
  }

  // setup the kernel run... the tensors will be filled out later in the gpu_memory
  ret->run_me = std::make_shared<kernel_run_t>(*reduce_cmd.run_me);
  ret->run_me->inputs.resize(reduce_cmd.run_me->inputs.num_args());
  ret->run_me->outputs.resize(reduce_cmd.run_me->outputs.num_args());
  ret->run_me->params = reduce_cmd.run_me->params;
  ret->run_me->ud = reduce_cmd.run_me->ud;

  return std::move(ret);
}
kernel_prep_ptr_t gpu_heuristic_t::_create_apply(command_id_t cmd) {

  auto &apply_cmd = apply_cmds[cmd];

  // fill out the stuff
  auto ret = std::make_shared<kernel_prep_t>();

  // fill it out
  ret->command_id = apply_cmd.id;
  ret->type = command_t::APPLY;
  ret->cpu_done = false;
  ret->gpu_done = false;
  ret->cpu_transfers = {};
  ret->gpu_transfers = {};
  ret->input = apply_cmd.input_tids;
  ret->input_sizes = apply_cmd.input_sizes;
  ret->output = apply_cmd.output_tids;
  ret->output_sizes = apply_cmd.output_sizes;
  ret->run_me = apply_cmd.run_me;

  return std::move(ret);
}
std::tuple<kernel_prep_ptr_t, int32_t>
gpu_heuristic_t::get_next_on_same(int32_t preffered_dev) {

  // first check if we have a reduce as we give priority to reduce
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    auto cur_dev = (preffered_dev + dev) % num_devices;
    if (!on_reduce_single_gpu[cur_dev].empty()) {

      // prepare the reduce kenel since we found on an op that can run it
      auto cmd = *on_reduce_single_gpu[cur_dev].begin();
      auto ret = _create_reduce(cmd);
      ret->dev = cur_dev;

      // return what we have
      return {std::move(ret), cur_dev};
    }
  }

  // next check of apply
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    auto cur_dev = (preffered_dev + dev) % num_devices;
    if (!on_apply_single_gpu[cur_dev].empty()) {

      // get the apply
      auto cmd = *on_apply_single_gpu[cur_dev].begin();
      auto ret = _create_apply(cmd);
      ret->dev = cur_dev;

      // return what we have
      return {std::move(ret), cur_dev};
    }
  }
  return {nullptr, -1};
};


kernel_prep_ptr_t gpu_heuristic_t::get_next_on_any(int32_t preffered_dev) {

  // go through each device and try to find a reduce we can run
  for (int32_t dev = 0; dev < num_devices; ++dev) {

    // we start from the preffered device
    auto cur_dev = (preffered_dev + dev) % num_devices;
    if(reduce_gpu_goodness_heuristic[cur_dev].empty()) {
      continue;
    }

    // get it from the goodness heuristic
    auto cmd = std::get<0>(reduce_gpu_goodness_heuristic[cur_dev].begin()->second);
    return _create_reduce(cmd);
  }

  // go through each device and try to find an apply 
  for (int32_t dev = 0; dev < num_devices; ++dev) {

    auto cur_dev = (preffered_dev + dev) % num_devices;
    if(apply_gpu_goodness_heuristic[cur_dev].empty()) {
      continue;
    }

    if (!apply_in_gpu_memory.empty()) {
      auto cmd = *apply_in_gpu_memory.begin();
      return _create_apply(cmd);
    }
  }

  return nullptr;
};

void gpu_heuristic_t::mark_as_scheduled(const kernel_prep_ptr_t &prep) {

  prep->kernel_prep_id = cur_prep_id++;
  auto cmd = prep->command_id;

  // make sure we are only dealing with APPLY and REDUCE commands
  assert(prep->type == command_t::APPLY || prep->type == command_t::REDUCE);
  if (prep->type == command_t::APPLY) {

    // remove it from the goodness heuristic in necessary
    auto apply_cmd_it = apply_cmds.find(cmd);
    if(apply_cmd_it->second.it != goodness_heuristic.end()) {
      goodness_heuristic.erase(apply_cmd_it->second.it);
    }

    // remove them from all the schedulings
    _apply_in_gpu_memory_remove(cmd);
    apply_cmds.erase(apply_cmd_it);
    apply_in_gpu_memory.erase(cmd);
    for (auto &asg : on_apply_single_gpu) {
      asg.erase(cmd);
    }

    // unlink all the inputs to this command
    for (auto in : prep->input) {
      _unlink_command_from_tensor(in, cmd);
    }

    // update the heuristic for all the inputs
    _update_heuristic_for_inputs(prep->input);

  } else {

    // we are issuing this one
    auto &reduce_op = reduce_cmds[cmd]; 
    reduce_op.num_issued++;

    // 1. remove it from the reduce heuristic
    if(reduce_op.it != goodness_heuristic.end()) {
      goodness_heuristic.erase(reduce_op.it);
      reduce_op.it = goodness_heuristic.end();
    }

    // 2. go through the inputs and remove them
    for (auto in : prep->input) {
      
      // try to find the inputs in the GPU if it is there remove it
      auto it = std::find(reduce_op.gpu_inputs.begin(),
                          reduce_op.gpu_inputs.end(), in);
      if(it != reduce_op.gpu_inputs.end()) {
        std::iter_swap(reduce_op.gpu_inputs.end() - 1, it);
        reduce_op.gpu_inputs.pop_back();
        continue;
      }
      
      // try to find the inputs in the CPU if it is there remove it
      auto jt = std::find(reduce_op.cpu_inputs.begin(),
                          reduce_op.cpu_inputs.end(), in);
      if(jt != reduce_op.cpu_inputs.end()) {
        std::iter_swap(reduce_op.cpu_inputs.end() - 1, jt);
        reduce_op.cpu_inputs.pop_back();
        continue;
      }

      // this should not happen
      assert(it != reduce_op.gpu_inputs.end() || 
             jt != reduce_op.cpu_inputs.end());
    }

    // 3.1. if we don't have anything else to run unschedule it
    if (reduce_op.gpu_inputs.size() < 2) {
      reduce_in_gpu_memory.erase(cmd);
      _reduce_in_gpu_memory_remove(cmd);
    }

    // 3.2. we also have to do this on per device basis
    for(int32_t dev = 0; dev < num_devices; ++dev) {

      // go through all the inputs try to find them per device...
      auto &dev_inputs = reduce_op.inputs_on_devices[dev];
      for (auto in : prep->input) {

        // find the inputs to the reduce and remove them
        auto it = std::find(dev_inputs.begin(), dev_inputs.end(), in);
        if(it == dev_inputs.end()) { continue;}

        std::iter_swap(dev_inputs.end() - 1, it);
        dev_inputs.pop_back();
      }

      // do we need to unschedule it per device
      if(dev_inputs.size() < 2) {
        on_reduce_single_gpu[dev].erase(cmd);
      }
    }

    // 4. check if we are done with the reduce
    if(reduce_op.num_issued == reduce_op.num_inputs - 1) {
      
      // remove them from all the schedulings
      reduce_cmds.erase(cmd);
    }

    // 5. unlink all the inputs to this command
    for (auto in : prep->input) {
      _unlink_command_from_tensor(in, cmd);
    }

    // 6. is this an intermediate result
    if(prep->output.front() < 0) {

      // 6.1. link it to this reduce
      tensors_to_cmds.insert({prep->output.front(), {reduce_op.id, command_t::REDUCE}});

      // 6.2. we are just scheduling the command therefore it must not exist before
      assert(tensors.find(prep->output.front()) == tensors.end());
      tensors[prep->output.front()] = {};

      // delete the intermediate result once it is not necessart
      tensors[prep->output.front()].should_delete = true;

      // 6.3. update the heuristic if necessary
      _update_heuristic_for_reduce(reduce_op.id);
      _update_gpu_heuristic_for_reduce(reduce_op.id);

      // 6.4. we use this anoymous tid
      inner_anon_id--;
    }
  }
}

bool gpu_heuristic_t::has_something() {
  return !reduce_cmds.empty() || !apply_cmds.empty();
}

void gpu_heuristic_t::remove_tensor(tid_t id) {

  //  if we can delete it immediately we do
  auto range = tensors_to_cmds.equal_range(id);
  if(std::distance(range.first, range.second) == 0) {
    tensors.erase(id);
    return;
  }

  // otherwise just mark it for removal
  auto it = tensors.find(id);
  it->second.should_delete = true;
}

void gpu_heuristic_t::_unlink_command_from_tensor(tid_t id, command_id_t cmd) {

  // go through all of them
  auto range = tensors_to_cmds.equal_range(id);
  auto num_left = std::distance(range.first, range.second);
  for (auto it = range.first; it != range.second; it++) {
    if (std::get<0>(it->second) == cmd) {

      // unlink it here
      tensors_to_cmds.erase(it);

      // remove the tensor if there is no other command using it
      // and we were tasked with removing it as soon as it is not used anymore
      auto t = tensors.find(id); assert(t != tensors.end());
      if(num_left == 1 && t->second.should_delete) { tensors.erase(t); }
      return;
    }
  }
  std::cerr << "This is not supposed to happen!";
  exit(-1);
}

void gpu_heuristic_t::_update_gpu_heuristic_for_inputs(const std::vector<tid_t> &inputs) {

  // update the heuristic for the commands that share inputs with it
  for (int32_t idx = 0; idx < inputs.size(); ++idx) {

    // get the input tid
    auto in_tid = inputs[idx];
    auto range = tensors_to_cmds.equal_range(in_tid);

    for(auto it = range.first; it != range.second; ++it) {

      // we just updated this command 
      auto [command, type] = it->second;

      // make sure it is the one of the two types of commands
      assert(type == command_t::APPLY || type == command_t::REDUCE);
      if(type == command_t::APPLY) {
        _update_gpu_heuristic_for_apply(command);
      }
      else {
        _update_gpu_heuristic_for_reduce(command);
      }
    }
  }
}

void gpu_heuristic_t::_update_gpu_heuristic_for_apply(command_id_t id) {
  _apply_in_gpu_memory_remove(id);
  _apply_in_gpu_memory_insert(id);
}

void gpu_heuristic_t::_apply_in_gpu_memory_insert(command_id_t cmd) {

  // make sure we have the inptus in GPU memory
  auto &c = apply_cmds[cmd];
  if(c.loaded_inputs != c.num_inputs) { return; }

  for(auto dev = 0; dev < num_devices; ++dev) {

    // calculate the right values
    auto inputs_left = c.num_inputs - c.inputs_on_devices[dev];
    auto num_used = _calculate_gpu_heuristic_apply(c.input_tids, dev);

    // insert it
    c.jts[dev] = apply_gpu_goodness_heuristic[dev].insert({{inputs_left, num_used}, {cmd, command_t::APPLY}});
  }
}

void gpu_heuristic_t::_apply_in_gpu_memory_remove(command_id_t cmd) {
  auto it = apply_cmds.find(cmd);
  for(auto dev = 0; dev < num_devices; ++dev) {
    if(it->second.jts[dev] == apply_gpu_goodness_heuristic[dev].end()) { continue; }
    apply_gpu_goodness_heuristic[dev].erase(it->second.jts[dev]);
    it->second.jts[dev] = apply_gpu_goodness_heuristic[dev].end();
  }
}

void gpu_heuristic_t::_update_gpu_heuristic_for_reduce(command_id_t id) {
  _reduce_in_gpu_memory_remove(id);
  _reduce_in_gpu_memory_insert(id);
}

void gpu_heuristic_t::_reduce_in_gpu_memory_insert(command_id_t cmd) {

  // get the command and make sure there are at least two inputs in GPU memory
  auto &c = reduce_cmds[cmd];
  if(c.gpu_inputs.size() < 2) { return; }

  for(auto dev = 0; dev < num_devices; ++dev) {

    // figure out if we have enough inputs
    auto needed_inputs = 0;
    for (auto &in : c.gpu_inputs) { 
      needed_inputs += tensors[in].on_device[dev]; 
      if(needed_inputs == 2) {  break; }
    }

    // calculate the heuristic
    int32_t heuristic_val = _calculate_gpu_heuristic_reduce(c.gpu_inputs, dev);

    // insert it 
    c.jts[dev] = reduce_gpu_goodness_heuristic[dev].insert({{needed_inputs, heuristic_val}, {cmd, command_t::REDUCE}});
  }
}

void gpu_heuristic_t::_reduce_in_gpu_memory_remove(command_id_t cmd){
  auto &c = reduce_cmds[cmd];
  for(auto dev = 0; dev < num_devices; ++dev) {
    if(c.jts[dev] == reduce_gpu_goodness_heuristic[dev].end()) { continue; }
    reduce_gpu_goodness_heuristic[dev].erase(c.jts[dev]);
    c.jts[dev] = reduce_gpu_goodness_heuristic[dev].end();
  }
}

kernel_prep_ptr_t gpu_heuristic_t::get_next_heuristic() { 

  if(goodness_heuristic.empty()) { return nullptr; }

  // get the thing from the goodness heuristic
  auto it = goodness_heuristic.begin();
  auto [cmd, type] = it->second;
  
  // check the type
  assert(type == command_t::APPLY ||  type == command_t::REDUCE);
  if(type == command_t::APPLY) { 
    return _create_apply(cmd); 
  }
  else { 
    return _create_reduce(cmd); 
  }
};

void gpu_heuristic_t::tensor_on_cpu(tid_t id) {

  // get the tensor
  auto &t =  tensors[id];
  
  // make sure the tensor was not already available
  bool is_new = !(t.on_cpu || t.gpu_copies > 0);

  // mark that is available
  assert(!t.on_cpu);
  t.on_cpu = true;

  // go and update the heuristic
  auto range = tensors_to_cmds.equal_range(id);
  for(auto it = range.first; it != range.second; ++it) {

    // we just updated this command 
    auto [command, type] = it->second;

    // make sure it is the one of the two types of commands
    assert(type == command_t::APPLY || type == command_t::REDUCE);
    if(type == command_t::APPLY) {

      // get the command
      auto &apply_cmd = apply_cmds[command];

      // if this is a new tensor make sure that is reflected and update the heuristic
      apply_cmd.inputs_available += is_new;
      _update_heuristic_for_apply(command);
      _update_gpu_heuristic_for_apply(command);
    }
    else {

      // if the tensor is only on the CPU
      if(t.gpu_copies == 0) {

        // get the command
        auto &reduce_cmd = reduce_cmds[command];

        // update the heuristic
        reduce_cmd.cpu_inputs.push_back(id);
        _update_heuristic_for_reduce(command);
        _update_gpu_heuristic_for_reduce(command);
      }
    }
  }
}

} // namespace bbts