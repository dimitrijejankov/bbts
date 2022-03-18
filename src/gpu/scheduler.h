#pragma once

#include "types.h"
#include "gpu_heuristic.h"
#include "gpu_memory.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"

namespace bbts {

class multi_gpu_scheduler_t {
public:
  multi_gpu_scheduler_t(size_t num_gpus, size_t gpu_mem_size, bbts::storage_ptr_t storage,
                        bbts::udf_manager_ptr udm, tensor_factory_ptr_t tf);

  // runs kernels that have the tensors in memory
  void gpu_execution_thread(int32_t dev);

  // moves the tensors from one gpu to another, this is usually faster
  void gpu_to_gpu_thread(int32_t dev);

  // CPU RAM to GPU RAM
  void cpu_to_gpu_thread();

  // moves memory from the CPU to GPUs
  void command_prep_thread();

  // reclaims memory on the GPU either by moving them back
  // to the CPU or straight up deleting them
  void gc_thread(int dev_id);

  void schedule_apply(bbts::command_ptr_t cmd);

  void schedule_reduce(bbts::command_ptr_t cmd);

  void mark_for_deletion(bbts::command_ptr_t cmd);

  void flush();

  void shutdown();

  // this schedules commands that are already known to be on the GPU
  gpu_heuristic_t heuristic;

  //
  gpu_memory_t mem;

  // we add kernels we have finished running here so that their stuff can be
  // unpinned
  scheduler_request_queue_t scheduler_queue;

  // we schedule the kernels in these queues
  std::vector<concurent_queue<kernel_prep_ptr_t>> run_queue;

  // we schedule all the requests for gpu to gpu transfers
  std::vector<concurent_queue<kernel_prep_ptr_t>> gpu2gpu_queue;

  // we schedule all the requests for cpu to gpu transfers
  concurent_queue<kernel_prep_ptr_t> cpu2gpu_queue;

  // we schedule here all the reqeusts for tensor garbage collection
  concurent_queue<gc_request_ptr_t> gc_queue;

  // the commands we can run immediately
  std::multimap<uint32_t, command_ptr_t> runnable_commands;

  // the meta data
  std::mutex meta_lck;
  std::unordered_map<tid_t, bbts::tensor_meta_t> meta;

  // the storage
  bbts::storage_ptr_t storage;

  // the manager
  bbts::udf_manager_ptr udm;

  // the tensorr factor
  tensor_factory_ptr_t tf;

  // by default the scheduler is running
  bool _is_running = true;

  // the number of GPUs in the system
  size_t _num_gpus;
};
using multi_gpu_scheduler_ptr_t = std::shared_ptr<multi_gpu_scheduler_t>;

} // namespace bbts