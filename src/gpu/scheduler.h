#pragma once

#include "types.h"
#include "gpu_heuristic.h"
#include "gpu_memory.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <cstddef>
#include <cstdint>

namespace bbts {

class multi_gpu_scheduler_t {
public:
  multi_gpu_scheduler_t(size_t num_gpus, size_t gpu_mem_size, bbts::storage_ptr_t storage,
                        bbts::udf_manager_ptr udm, tensor_factory_ptr_t tf);

  // mark that a tensor is created
  void mark_tensor_on_cpu(tid_t tid, size_t num_bytes, tensor_meta_t meta);

  // flush all the tensors currently residing exclusviely in the GPU memory into the CPU memory
  void flush();

  // shutdown the scheduler
  void shutdown();

  // returns the number of GPUs we are managing
  int32_t num_gpus() const;

  // schedule a bunch of commands
  void schedule(std::vector<bbts::command_ptr_t> &to_schedule);

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

private: 

  // schedule an apply to be run on the GPU
  gpu_command_schedule_ptr_t _prepare_apply(bbts::command_ptr_t &cmd);

  // schedule a reduce to be run on the GPU
  gpu_command_schedule_ptr_t _prepare_reduce(bbts::command_ptr_t &cmd);

  // mark that this tensor is not necessary anymore and can be safely deleted
  gpu_command_schedule_ptr_t _prepare_delete(bbts::command_ptr_t &cmd);

  // performs the actual flushing
  void _perform_flush();

  // performs the actual shutdown
  void _perform_shutdown();

  // schedule a kernel prep for execution, this could 
  // involve preallocating memory as well as issuing a grabage collection request
  bool _schedule_for_execution(kernel_prep_ptr_t prep, int32_t dev);

  // the device where we prefer for the kernel to be launched
  int preffered_dev = 0;

  // this schedules commands that are already known to be on the GPU
  gpu_heuristic_t heuristic;

  // the gpu memory
  gpu_memory_t mem;

  // we add kernels we have finished running here 
  // so that their stuff can be unpinned
  scheduler_request_queue_t scheduler_queue;

  // all the outstanding flush requests
  std::vector<flush_request_ptr_t> outstanding_flush_requests;

  // we schedule the kernels in these queues
  std::vector<concurent_queue<kernel_prep_ptr_t>> run_queue;

  // we schedule all the requests for gpu to gpu transfers
  std::vector<concurent_queue<kernel_prep_ptr_t>> gpu2gpu_queue;

  // we schedule all the requests for cpu to gpu transfers
  concurent_queue<kernel_prep_ptr_t> cpu2gpu_queue;

  // we schedule here all the reqeusts for tensor garbage collection
  std::vector<concurent_queue<gc_request_ptr_t>> gc_queue;

  // the commands we can run immediately
  std::multimap<uint32_t, command_ptr_t> runnable_commands;

  // the meta data
  std::mutex meta_lck;
  std::unordered_map<tid_t, bbts::tensor_meta_t> _meta;

  // the anon tensor mapping from the GPU memory to RAM
  std::mutex anon_lck;
  std::unordered_map<tid_t, tid_t> _anon_gpu_cpu;
  std::unordered_map<tid_t, tid_t> _anon_cpu_gpu;

  // the storage
  bbts::storage_ptr_t storage;

  // the manager
  bbts::udf_manager_ptr udm;

  // the tensorr factor
  tensor_factory_ptr_t tf;

  // the number of GPUs in the system
  size_t _num_gpus;

  // the number of unfinished kernels
  size_t num_unfinished_kernels;
};
using multi_gpu_scheduler_ptr_t = std::shared_ptr<multi_gpu_scheduler_t>;

} // namespace bbts