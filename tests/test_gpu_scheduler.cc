#include "../src/gpu/scheduler.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mkl_types.h>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std::chrono;

std::vector<std::thread>
run_threads(bbts::multi_gpu_scheduler_ptr_t scheduler) {

  std::vector<std::thread> threads;
  threads.push_back(
      std::thread([scheduler]() { scheduler->command_prep_thread(); }));

  threads.push_back(
      std::thread([scheduler]() { scheduler->cpu_to_gpu_thread(); }));

  for (auto dev = 0; dev < scheduler->_num_gpus; ++dev) {

    threads.push_back(std::thread([scheduler, dev]() { 
      scheduler->reaper_thread(dev); 
    }));

    threads.push_back(std::thread([scheduler, dev]() { 
      scheduler->gpu_execution_thread(dev); 
    }));

    threads.push_back(std::thread([scheduler, dev]() { 
      scheduler->gpu_to_gpu_thread(dev);
    }));
  }

  return std::move(threads);
}

int main() {

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      4, storage, manager, factory);

  // try to deserialize
  // uniform 0
  // uniform 1
  // ...
  // uniform n
  bbts::parsed_command_list_t gen_cmd_list;
  bool success = gen_cmd_list.deserialize("gen.bbts");

  // try to deserialize
  // mult 0 2
  // ...
  // mult 0 2
  // reduce ...
  // delete ...
  bbts::parsed_command_list_t run_cmd_list;
  success = run_cmd_list.deserialize("run.bbts");

  // compile all the commands
  bbts::command_loader_t compiler(*factory, *manager);
  auto gen_cmds = compiler.compile(gen_cmd_list);
  auto run_cmds = compiler.compile(gen_cmd_list);

  // schedule the apply
  for (auto &cmd : gen_cmds) {

    if (cmd->is_apply()) {
      scheduler->schedule_apply(std::move(cmd));
    } else {
      throw std::runtime_error("not supposed to happen!");
    }
  }

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // schedule the run commands
  for (auto &cmd : run_cmds) {
    if (cmd->is_apply()) {
      scheduler->schedule_apply(std::move(cmd));
    } else if (cmd->is_reduce()) {
      scheduler->schedule_reduce(std::move(cmd));
    } else if (cmd->is_delete()) {
      scheduler->mark_for_deletion(std::move(cmd));
    } else {
      throw std::runtime_error("not supposed to happen!");
    }
  }

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  return 0;
}