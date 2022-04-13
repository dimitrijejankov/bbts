#pragma once

#include "../commands/command_loader.h"
#include "../commands/parsed_command.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include "../utils/concurent_queue.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>


namespace bbts {

#define BBTS_MAX_GPU_DEVICES 8

// the flush request
struct flush_request_t {
  
  // we fill this out once the request is done
  std::promise<bool> status;
};
using flush_request_ptr_t = std::shared_ptr<flush_request_t>; 

struct kernel_run_t {

  // the ud function of the kernel we want to run
  ud_impl_t *ud;

  // the inputs for the kernel
  ud_impl_t::tensor_args_t inputs;

  // the outputs for the kernel
  ud_impl_t::tensor_args_t outputs;

  // the parameters for the kernel
  bbts::ud_impl_t::tensor_params_t params;
};
using kernel_run_ptr_t = std::shared_ptr<kernel_run_t>;

// the id of both GPU2GPU and CPU2GPU transfers
using transfer_id_t = int64_t;

struct cpu_to_gpu_transfer_t {

  // the id of the transfer so that we can easly match it
  transfer_id_t id;

  // locks the cpu2gpu transfer
  std::mutex m;

  // is the transfer finished
  bool is_finished = false;

  // the tid of the tensor
  tid_t tid;

  // the size in bytes
  size_t num_bytes;

  // the destination on the GPU
  tensor_t *dst; 

  // the destination device
  int32_t dst_dev;
};
using cpu_to_gpu_transfer_ptr_t = std::shared_ptr<cpu_to_gpu_transfer_t>;


struct gpu_to_gpu_transfer_t {

  // the id of the transfer so that we can easly match it
  transfer_id_t id;

  // the tid of the tensor we are transfering
  tid_t tid;

  // the source device
  int32_t src_dev;

  // the source from the CPU 
  tensor_t *src;

  // the destination on the GPU
  tensor_t *dst; 

  // the destination device
  int32_t dst_dev;

  // the number of bytes
  size_t num_bytes;

  // this wil be set to a cpu to gpu transfer if 
  cpu_to_gpu_transfer_ptr_t depends;

  // is this finished
  bool is_finished = false;
};
using gpu_to_gpu_transfer_ptr_t = std::shared_ptr<gpu_to_gpu_transfer_t>;

struct kernel_prep_t {

  // the id of the command the kernel is associated with
  command_id_t command_id;

  // the type of the command the kernel comes from
  bbts::command_t::op_type_t type;

  // we want to run this kernel but we first need to prepare it
  kernel_run_ptr_t run_me;

  // the device where we are scheduling it
  int32_t dev;

  // the inputs
  std::vector<tid_t> input;

  // the input sizes in bytes
  std::vector<size_t> input_sizes;

  // the outputs that were created
  std::vector<tid_t> output;

  // the output sizes in bytes
  std::vector<size_t> output_sizes;

  // all the CPU transfers we need to preform
  std::vector<cpu_to_gpu_transfer_ptr_t> cpu_transfers;

  // all the GPU transfers we need to do
  std::vector<gpu_to_gpu_transfer_ptr_t> gpu_transfers;

  // lock this so we don't interfere with other scheduling the kernel
  std::mutex m;

  // are the gpu2gpu transfers finished
  bool gpu_done = false;

  // are the cpu2gpu transfers finished
  bool cpu_done = false;
};
using kernel_prep_ptr_t = std::shared_ptr<kernel_prep_t>;

struct gc_request_t {

  // the device for which the request is for
  int32_t dev;

  // list of tensors we are supposed to free
  std::vector<std::tuple<tensor_t *, tid_t, size_t>> to_free;

  // list of tensors we are supposed to evict
  std::vector<std::tuple<tensor_t *, tid_t, size_t>> to_evict;

  // tensors that are to be unpinned once the request is finished
  // this is unpinned by the GPU memory
  std::vector<std::tuple<tid_t, size_t>> to_unpin;

  // the kernel prep to run once the request is finished
  kernel_prep_ptr_t to_run;
};
using gc_request_ptr_t = std::shared_ptr<gc_request_t>;

struct gpu_command_schedule_t {

  // the function we want to run
  ud_impl_t *fn;

  // the number of bytes each input has
  std::vector<size_t> input_sizes;

  // the number of bytes each output has
  std::vector<size_t> output_sizes;

  // the parameters
  bbts::ud_impl_t::tensor_params_t params;

  // the command
  bbts::command_ptr_t cmd;
};
using gpu_command_schedule_ptr_t = std::shared_ptr<gpu_command_schedule_t>;

struct scheduler_request_t {

  // list of all the kernels since the last time the thread was woken up
  std::vector<kernel_prep_ptr_t> retired_kernels;
  
  // the finished request for freeing tensors or evicting them
  std::vector<gc_request_ptr_t> finished_gc;

  // all the commands that were scheduled since the last time the thread was woken up
  std::vector<gpu_command_schedule_ptr_t> to_schedule;

  // all the cpu to gpu transfers that were scheduled since the last time the thread was woken up
  std::vector<cpu_to_gpu_transfer_ptr_t> cpu_transfers;

  // all the gpu to gpu transfers that were scheduled since the last time the thread was woken up
  std::vector<gpu_to_gpu_transfer_ptr_t> gpu_transfers;

  // all the flush requests that were made since the last time the thread was woken up
  std::vector<flush_request_ptr_t> flush_requests;

  // all the tensors that were created on the CPU in the mean time
  std::vector<std::tuple<tid_t, size_t>> cpu_created_tensors; 

  // is the scheduler shutdown
  bool shutdown = false;

  // clear all the stuff
  void clear() {
    retired_kernels.clear();
    finished_gc.clear();
    to_schedule.clear();
    cpu_transfers.clear();
    gpu_transfers.clear();
    flush_requests.clear();
    cpu_created_tensors.clear();
  }
};
using scheduler_request_ptr_t = std::shared_ptr<scheduler_request_t>;

class scheduler_request_queue_t {
public:

  // mark that the scheduler is done
  void signal_shutdown() {
    std::unique_lock<std::mutex> lck(m);
    shutdown = true;
    cv.notify_all();
  }

  // signal that the tensor is on the CPU
  void signal_tensor_on_cpu(tid_t tid, size_t num_bytes) {
    std::unique_lock<std::mutex> lck(m);
    cpu_created_tensors.push_back({tid, num_bytes});
    cv.notify_all();
  }

  // mark that the kernel is done
  void signal_kernel_done(kernel_prep_ptr_t kernel) {

    std::unique_lock<std::mutex> lck(m);
    retired_kernels.push_back(std::move(kernel));
    cv.notify_all();
  }

  // signal reaper done
  void signal_reaper_done(gc_request_ptr_t req) {

    std::unique_lock<std::mutex> lck(m);
    finished_gc.push_back(std::move(req));
    cv.notify_all();
  }

  // signal that new commands have arrived
  void signal_cmds_scheduled(std::vector<gpu_command_schedule_ptr_t> cmds) {

    std::unique_lock<std::mutex> lck(m);
    for(auto &cmd : cmds) {
      to_schedule.push_back(std::move(cmd));
    }
    new_commands = true;
    cv.notify_all();
  }

  void signal_cpu_to_gpu_transfer_done(std::vector<cpu_to_gpu_transfer_ptr_t> &transfer) {

    std::unique_lock<std::mutex> lck(m);

    // move them here
    for(auto &tr : transfer) {
      cpu_transfers.push_back(std::move(tr));
    }
    cv.notify_all();
  }

  std::future<bool> signal_flush_request() {
    
    std::unique_lock<std::mutex> lck(m);
    flush_requests.push_back(std::make_shared<flush_request_t>());
    cv.notify_all();
    return flush_requests.back()->status.get_future();
  }

  void signal_gpu_to_gpu_transfer_done(std::vector<gpu_to_gpu_transfer_ptr_t> &transfer){

    std::unique_lock<std::mutex> lck(m);

    // move the transfers here
    for(auto &tr : transfer) {
      gpu_transfers.push_back(std::move(tr));
    }
    cv.notify_all();
  }

  void wait_dequeue(scheduler_request_ptr_t &req) {

    // wait to get something
    std::unique_lock<std::mutex> lck(m);
    cv.wait(lck, [&] {
      return (!retired_kernels.empty() || 
              !finished_gc.empty() ||
              !to_schedule.empty() ||
              shutdown);
    });

    // clear it just in case
    req->clear();

    // give the updates
    std::swap(req->retired_kernels, retired_kernels);
    std::swap(req->finished_gc, finished_gc);
    std::swap(req->to_schedule, to_schedule);
    std::swap(req->cpu_transfers, cpu_transfers);
    std::swap(req->gpu_transfers, gpu_transfers);
    std::swap(req->flush_requests, flush_requests);
    std::swap(req->cpu_created_tensors, cpu_created_tensors);

    // forward the shutdown request if any
    req->shutdown = shutdown;
  }

  bool has_something() {

    std::unique_lock<std::mutex> lck(m);
    return (!retired_kernels.empty() || 
            !finished_gc.empty() ||
            !to_schedule.empty() || 
            !flush_requests.empty() ||
            shutdown);
  }

private:

  bool new_commands = false;

  bool shutdown = false;

  std::vector<kernel_prep_ptr_t> retired_kernels;

  std::vector<gc_request_ptr_t> finished_gc;

  std::vector<gpu_command_schedule_ptr_t> to_schedule;

  std::vector<cpu_to_gpu_transfer_ptr_t> cpu_transfers;

  std::vector<gpu_to_gpu_transfer_ptr_t> gpu_transfers;

  std::vector<flush_request_ptr_t> flush_requests;

  std::vector<std::tuple<tid_t, size_t>> cpu_created_tensors;

  // locks the scheduler queue
  std::mutex m;

  // used for waking up and signaling
  std::condition_variable cv;
};
using scheduler_request_queue_ptr_t =
    std::shared_ptr<scheduler_request_queue_t>;

} // namespace bbts
