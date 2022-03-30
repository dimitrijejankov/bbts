#pragma once

#include "../commands/command_loader.h"
#include "../commands/parsed_command.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include "../utils/concurent_queue.h"
#include <cstddef>
#include <cstdint>
#include <memory>


namespace bbts {

#define BBTS_MAX_GPU_DEVICES 8

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

  // the kernel prep to run once the request is finished
  kernel_prep_ptr_t to_run;
};
using gc_request_ptr_t = std::shared_ptr<gc_request_t>;

struct apply_schedule_t {

  // the function we want to run
  ud_impl_t *fn;

  // the number of bytes each input has
  std::vector<size_t> input_sizes;

  // the number of bytes each output has
  std::vector<size_t> output_sizes;

  // the command
  bbts::command_ptr_t cmd;
};
using apply_schedule_ptr_t = std::shared_ptr<apply_schedule_t>;

struct reduce_schedule_t {

  // the function we want to run
  ud_impl_t *fn;

  // the parameters of the UD function we want to run
  ud_impl_t::tensor_params_t _params;

  // the number of bytes each input has
  std::vector<size_t> input_sizes;

  // the number of bytes each output has
  size_t output_size;

  // the command
  bbts::command_ptr_t cmd;
};
using reduce_schedule_ptr_t = std::shared_ptr<reduce_schedule_t>;

struct delete_schedule_t {

  // the command
  bbts::command_ptr_t cmd;
};
using delete_schedule_ptr_t = std::shared_ptr<delete_schedule_t>;

struct scheduler_request_t {

  // list of all the kernels since the last time the thread was woken up
  std::vector<kernel_prep_ptr_t> retired_kernels;
  
  // the finished request for freeing tensors or evicting them
  std::vector<gc_request_ptr_t> finished_gc;

  // all the applies that were scheduled since the last time the thread was woken up
  std::vector<apply_schedule_ptr_t> apply_cmds;

  // all the reduces that were scheduled since the last time the thread was woken up
  std::vector<reduce_schedule_ptr_t> reduce_cmds;

  // all the cpu to gpu transfers that were scheduled since the last time the thread was woken up
  std::vector<cpu_to_gpu_transfer_ptr_t> cpu_transfers;

  // all the gpu to gpu transfers that were scheduled since the last time the thread was woken up
  std::vector<gpu_to_gpu_transfer_ptr_t> gpu_transfers;
};
using scheduler_request_ptr_t = std::shared_ptr<scheduler_request_t>;

class scheduler_request_queue_t {
public:
  // mark that the kernel is done
  void signal_kernel_done(kernel_prep_ptr_t kernel) {

    std::unique_lock<std::mutex> lck;
    retired_kernels.push_back(std::move(kernel));
  }

  // signal reaper done
  void signal_reaper_done(gc_request_ptr_t req) {

    std::unique_lock<std::mutex> lck;
    finished_gc.push_back(std::move(req));
  }

  // signal new apply
  void signal_apply(apply_schedule_ptr_t apply) {

    std::unique_lock<std::mutex> lck;
    apply_cmds.push_back(std::move(apply));

    new_commands = true;
  }

  // signal new reduce
  void signal_reduce(reduce_schedule_ptr_t reduce) {

    std::unique_lock<std::mutex> lck;
    reduce_cmds.push_back(std::move(reduce));

    new_commands = true;
  }

  void signal_delete(delete_schedule_ptr_t reduce) {

    std::unique_lock<std::mutex> lck;
    delete_cmds.push_back(std::move(reduce));

    new_commands = true;
  }

  void signal_cpu_to_gpu_transfer_done(std::vector<cpu_to_gpu_transfer_ptr_t> &transfer) {

    std::unique_lock<std::mutex> lck;

    // move them here
    for(auto &tr : transfer) {
      cpu_transfers.push_back(std::move(tr));
    }
  }

  void signal_gpu_to_gpu_transfer_done(std::vector<gpu_to_gpu_transfer_ptr_t> &transfer){

    std::unique_lock<std::mutex> lck;

    // move the transfers here
    for(auto &tr : transfer) {
      gpu_transfers.push_back(std::move(tr));
    }
  }

  void wait_dequeue(scheduler_request_ptr_t &req) {

    // wait to get something
    std::unique_lock<std::mutex> lck;
    cv.wait(lck, [&] {
      return (!retired_kernels.empty() || !finished_gc.empty() ||
              !apply_cmds.empty() || !reduce_cmds.empty());
    });

    // give the updates
    std::swap(req->retired_kernels, retired_kernels);
    std::swap(req->finished_gc, finished_gc);
    std::swap(req->apply_cmds, apply_cmds);
    std::swap(req->reduce_cmds, reduce_cmds);
    std::swap(req->cpu_transfers, cpu_transfers);
    std::swap(req->gpu_transfers, gpu_transfers);
  }

  bool has_something() {

    std::unique_lock<std::mutex> lck;
    return (!retired_kernels.empty() || !finished_gc.empty() ||
            !apply_cmds.empty() || !reduce_cmds.empty());
  }

private:
  bool new_commands = false;

  std::vector<kernel_prep_ptr_t> retired_kernels;

  std::vector<gc_request_ptr_t> finished_gc;

  std::vector<apply_schedule_ptr_t> apply_cmds;

  std::vector<reduce_schedule_ptr_t> reduce_cmds;

  std::vector<delete_schedule_ptr_t> delete_cmds;

  std::vector<cpu_to_gpu_transfer_ptr_t> cpu_transfers;

  std::vector<gpu_to_gpu_transfer_ptr_t> gpu_transfers;

  // locks the scheduler queue
  std::mutex m;

  // used for waking up and signaling
  std::condition_variable cv;
};
using scheduler_request_queue_ptr_t =
    std::shared_ptr<scheduler_request_queue_t>;

} // namespace bbts
