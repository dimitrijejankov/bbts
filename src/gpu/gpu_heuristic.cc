#include "gpu_heuristic.h"
#include <cassert>
#include <tuple>

namespace bbts {

gpu_heuristic_t::gpu_heuristic_t(uint32_t num_devices)
    : num_devices(num_devices) {

  on_apply_single_gpu.resize(num_devices);
  on_reduce_single_gpu.resize(num_devices);
}
void gpu_heuristic_t::tensor_loaded(tid_t id, int dev) {

  // mark that the tensor is now on the device
  auto &t = tensors[id];
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
      apply_cmds[command_id].inputs_on_devices[dev]++;
      apply_cmds[command_id].loaded_inputs += t.gpu_copies == 1;

      // can we schedule it
      if (apply_cmds[command_id].loaded_inputs ==
          apply_cmds[command_id].num_inputs) {
        apply_in_gpu_memory.insert(command_id);
      }

      // can we schedule it on a single device
      if (apply_cmds[command_id].inputs_on_devices[dev] ==
          apply_cmds[command_id].num_inputs) {
        on_apply_single_gpu[dev].insert(command_id);
      }
    } else if (command_type == command_t::REDUCE) {

      // mark that we have it now on the device
      auto &cmd = reduce_cmds[command_id];
      cmd.inputs_on_devices[dev].push_back(id);

      // add this input if we don't already have it
      auto it =
          std::find(cmd.loaded_inputs.begin(), cmd.loaded_inputs.end(), id);
      if (it == cmd.loaded_inputs.end()) {
        cmd.loaded_inputs.push_back(id);
      }

      // if there are at least two inputs of a reduce it can be scheduled
      if (cmd.loaded_inputs.size() == 2) {
        reduce_in_gpu_memory.insert(command_id);
      }

      // same is true when we look at the per GPU case
      if (cmd.inputs_on_devices[dev].size() == 2) {
        on_reduce_single_gpu[dev].insert(command_id);
      }
    }
  }

  // update the heuristic since we got the first copy in GPU memory
  if (t.gpu_copies == 1) {
    // tensor_update_heuristic(id, true);
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
      apply_cmds[command_id].inputs_on_devices[dev]--;
      apply_cmds[command_id].loaded_inputs -= t.gpu_copies == 0;

      // should we unschedule it
      if (apply_cmds[command_id].loaded_inputs !=
          apply_cmds[command_id].num_inputs) {
        apply_in_gpu_memory.erase(command_id);
      }

      // should we unschedule it from a particular device
      if (apply_cmds[command_id].inputs_on_devices[dev] !=
          apply_cmds[command_id].num_inputs) {
        on_apply_single_gpu[dev].erase(command_id);
      }
    } else if (command_type == command_t::REDUCE) {

      // get the command
      auto &cmd = reduce_cmds[command_id];

      // if this is the last copy of the tensor
      if (t.gpu_copies == 0) {

        // find it the tensor (not finding it is a bug)
        auto it =
            std::find(cmd.loaded_inputs.begin(), cmd.loaded_inputs.end(), id);
        assert(it != cmd.loaded_inputs.end());

        // remove it
        std::swap(*it, *(cmd.loaded_inputs.end() - 1));
        cmd.loaded_inputs.pop_back();

        // if we dropped to one input we need to unschedule it
        if (cmd.loaded_inputs.size() == 1) {
          reduce_in_gpu_memory.erase(command_id);
        }
      }

      // find the tensor and remove it, not finding it is a bug
      auto &dev_inputs = cmd.inputs_on_devices[dev];
      auto it = std::find(dev_inputs.begin(), dev_inputs.end(), id);
      std::swap(*it, *(dev_inputs.end() - 1));
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
uint64_t gpu_heuristic_t::calculate_heuristic(const std::vector<tid_t> inputs) {

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
void gpu_heuristic_t::tensor_update_heuristic(tid_t id, bool is_loaded) {

  auto range = tensors_to_cmds.equal_range(id);
  auto num_affected = std::distance(range.first, range.second);
  for (auto it = range.first; it != range.second; ++it) {

    auto [cmd, type] = it->second;
    if (cmd == command_t::op_type_t::APPLY) {

      // check if we even have all the inputs to run the appl
      auto &apply_cmd = apply_cmds[cmd];
      if (apply_cmd.inputs_available == apply_cmd.num_inputs) {

        // make sure we have the iterator
        assert(apply_cmd.it != goodness_heuristic.end());

        // calculate the new values
        auto [num_needed, heuristic] = apply_cmd.it->first;
        num_needed += is_loaded ? 1 : -1;
        heuristic += is_loaded ? -num_affected : num_affected;

        // remove it from the heuristic
        goodness_heuristic.erase(apply_cmd.it);

        // update the goodness heuristic
        apply_cmd.it =
            goodness_heuristic.insert({{num_needed, heuristic}, cmd});
      }
    } else if (cmd == command_t::op_type_t::REDUCE) {

      // check if we have all the inputs
      auto &reduce_cmd = reduce_cmds[cmd];
      if (reduce_cmd.cpu_inputs.size() + reduce_cmd.loaded_inputs.size() == 2) {
        continue;
      }

      // make sure we have the iterator
      assert(reduce_cmd.it != goodness_heuristic.end());

      // get the previous numbers
      auto [num_needed, heuristic] = reduce_cmd.it->first;

      // we only need two inputs so anything about that
      if (reduce_cmd.loaded_inputs.size() == 0) {
        num_needed = 2;
        heuristic = std::get<1>(reduce_cmd.cpu_inputs[0]) +
                    std::get<1>(reduce_cmd.cpu_inputs[1]);
      } else if (reduce_cmd.loaded_inputs.size() == 1) {
        num_needed = 1;
        heuristic = std::get<1>(reduce_cmd.cpu_inputs[0]);
      } else {
        num_needed = 0;
        heuristic = 0;
      }

      // remove it from the heuristic
      goodness_heuristic.erase(reduce_cmd.it);

      // update the goodness heuristic
      reduce_cmd.it = goodness_heuristic.insert({{num_needed, heuristic}, cmd});
    }
  }
}
void gpu_heuristic_t::register_apply(bbts::apply_schedule_ptr_t &apply_sch) {

  auto &cmd = apply_sch->cmd;
  auto &apply_cmd = apply_cmds[cmd->id];

  apply_cmd.id = cmd->id;
  apply_cmd.num_inputs = cmd->get_num_inputs();
  apply_cmd.inputs_available = 0;
  apply_cmd.loaded_inputs = 0;
  apply_cmd.input_tids.reserve(cmd->get_num_inputs());
  apply_cmd.output_tids.reserve(cmd->get_num_outputs());

  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    auto in_tid = cmd->get_inputs()[idx].tid;
    auto &in = tensors[in_tid];

    // do we have it on any of the GPUs
    if (in.gpu_copies > 0) {
      apply_cmd.loaded_inputs++;
    }

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

  // store the output tids
  for (int32_t idx = 0; idx < cmd->get_num_outputs(); ++idx) {
    apply_cmd.output_tids.push_back(cmd->get_outputs()[idx].tid);
  }

  // check if we have enough inputs on any GPU
  if (apply_cmd.loaded_inputs == apply_cmd.num_inputs) {
    apply_in_gpu_memory.insert(apply_cmd.id);
  }

  // check if we have enough inputs per GPU
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    if (apply_cmd.inputs_on_devices[dev] == num_devices) {
      on_apply_single_gpu[dev].insert(apply_cmd.id);
    }
  }
};
void gpu_heuristic_t::register_reduce(bbts::reduce_schedule_ptr_t &reduce_sch) {

  auto &cmd = reduce_sch->cmd;
  auto &reduce_cmd = reduce_cmds[cmd->id];

  reduce_cmd.id = cmd->id;
  reduce_cmd.num_inputs = cmd->get_num_inputs();
  reduce_cmd.num_issued = 0;

  for (int32_t idx = 0; idx < cmd->get_num_inputs(); ++idx) {

    auto in_tid = cmd->get_inputs()[idx].tid;
    auto &in = tensors[in_tid];

    // do we have it on any of the GPUs
    if (in.gpu_copies > 0) {
      reduce_cmd.loaded_inputs.push_back(in_tid);
    }

    // update the per device count
    for (int32_t dev = 0; dev < num_devices; ++dev) {
      if (in.on_device[dev]) {
        reduce_cmd.inputs_on_devices[dev].push_back(in_tid);
      }
    }

    // add a link
    tensors_to_cmds.insert({in_tid, {reduce_cmd.id, command_t::REDUCE}});
  }

  // check if we have enough inputs on any GPU
  if (reduce_cmd.loaded_inputs.size() >= 2) {
    reduce_in_gpu_memory.insert(reduce_cmd.id);
  }

  // check if we have enough inputs per GPU
  for (int32_t dev = 0; dev < num_devices; ++dev) {
    if (reduce_cmd.inputs_on_devices[dev].size() >= 2) {
      on_reduce_single_gpu[dev].insert(reduce_cmd.id);
    }
  }
};
kernel_prep_ptr_t gpu_heuristic_t::create_reduce(command_id_t cmd) {

  // grab the reduce
  auto &reduce_cmd = reduce_cmds[cmd];

  // fill it out
  auto ret = std::make_shared<kernel_prep_t>();
  ret->command_id = reduce_cmd.id;
  ret->type = command_t::REDUCE;
  ret->cpu_done = false;
  ret->gpu_done = false;
  ret->cpu_transfers = {};
  ret->gpu_transfers = {};
  ret->input = {reduce_cmd.loaded_inputs[0], reduce_cmd.loaded_inputs[1]};

  // check if this is the last time we are issuing a reduce
  if (reduce_cmd.num_issued == reduce_cmd.num_inputs - 1) {
    ret->output = {reduce_cmd.output_tid};
  } else {
    ret->output = {inner_anon_id++};

    // TODO linup stuff so that
  }

  ret->run_me = reduce_cmd.run_me;
  return std::move(ret);
}
kernel_prep_ptr_t gpu_heuristic_t::create_apply(command_id_t cmd) {

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
  ret->output = apply_cmd.output_tids;
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
      auto ret = create_reduce(cmd);
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
      auto ret = create_apply(cmd);
      ret->dev = cur_dev;

      // return what we have
      return {std::move(ret), cur_dev};
    }
  }
  return {nullptr, -1};
};

kernel_prep_ptr_t gpu_heuristic_t::get_next_on_any() {

  if (!reduce_in_gpu_memory.empty()) {
    auto cmd = *reduce_in_gpu_memory.begin();
    return create_apply(cmd);
  }

  if (!apply_in_gpu_memory.empty()) {
    auto cmd = *apply_in_gpu_memory.begin();
    return create_apply(cmd);
  }
  return nullptr;
};

void gpu_heuristic_t::mark_as_scheduled(const kernel_prep_ptr_t &prep) {

  auto cmd = prep->command_id;

  // make sure we are only dealing with APPLY and REDUCE commands
  assert(prep->type == command_t::APPLY || prep->type == command_t::REDUCE);
  if (prep->type == command_t::APPLY) {

    // remove them from all the schedulings
    apply_cmds.erase(cmd);
    apply_in_gpu_memory.erase(cmd);
    for (auto &asg : on_apply_single_gpu) {
      asg.erase(cmd);
    }

    // unlink all the inputs to this command
    for (auto in : prep->input) {
      _unlink_command_from_tensor(in, cmd);
    }
  } else {

    // remove them from all the schedulings
    reduce_cmds.erase(cmd);
    reduce_in_gpu_memory.erase(cmd);
    for (auto &rsg : on_reduce_single_gpu) {
      rsg.erase(cmd);
    }

    // unlink all the inputs to this command
    for (auto in : prep->input) {
      _unlink_command_from_tensor(in, cmd);
    }
  }
}

kernel_prep_ptr_t gpu_heuristic_t::get_next_heuristic() { return nullptr; };
void gpu_heuristic_t::tensor_available_update_heuristic(tid_t id) {}
void gpu_heuristic_t::tensor_on_cpu(tid_t id) {}

void gpu_heuristic_t::_unlink_command_from_tensor(tid_t id, command_id_t cmd) {
  auto range = tensors_to_cmds.equal_range(id);
  for (auto it = range.first; it != range.second; ++it) {
    if (std::get<0>(it->second) == cmd) {
      tensors_to_cmds.erase(it);
      return;
    }
  }
  std::cerr << "This is not supposed to happen!";
  exit(-1);
}
} // namespace bbts