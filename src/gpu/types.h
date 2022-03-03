#pragma once

#include "../commands/command_loader.h"
#include "../commands/parsed_command.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include "../utils/concurent_queue.h"


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

struct kernel_prep_t {

  // the id of the command the kernel is associated with
  int32_t command_id;

  // we want to run this kernel but we first need to prepare it
  kernel_run_ptr_t run_me;

  // the device where we are scheduling it
  int32_t dev;

  // the inputs
  std::vector<tid_t> input;

  // the outputs that were created
  std::vector<tid_t> output;

  // all the CPU transfers we need to preform (tid, input, num_bytes)
  std::vector<std::tuple<tid_t, uint32_t, size_t>> cpu_transfers;

  // all the GPU transfers we need to do (tensor, src_dev, input index,
  // num_bytes)
  std::vector<std::tuple<tensor_t *, int32_t, uint32_t, size_t>> gpu_transfers;

  // lock this so we don't interfere with other scheduling the kernel
  std::mutex m;

  // are the gpu2gpu transfers finished
  bool gpu_done = false;

  // are the cpu2gpu transfers finished
  bool cpu_done = false;
};
using kernel_prep_ptr_t = std::shared_ptr<kernel_prep_t>;

struct reaper_request_t {

  // list of tensors we are supposed to free
  std::vector<tensor_t *> to_free;

  // list of tensors we are supposed to evict
  std::vector<std::tuple<tensor_t *, tid_t, size_t>> to_evict;

  // the kernel prep to run once the request is finished
  kernel_prep_ptr_t to_run;
};
using reaper_request_ptr_t = std::shared_ptr<reaper_request_t>;

struct apply_schedule_t {

  // the function we want to run
  ud_impl_t *fn;

  // the number of bytes each input has
  std::vector<size_t> input_num_bytes;

  // the number of bytes each output has
  std::vector<size_t> output_num_bytes;

  // the command
  bbts::command_ptr_t cmd;
};
using apply_schedule_ptr_t = std::shared_ptr<apply_schedule_t>;

struct reduce_schedule_t {

  // the function we want to run
  ud_impl_t *fn;

  //
  ud_impl_t::tensor_params_t _params;

  // the input meta
  bbts::ud_impl_t::meta_args_t input_meta;

  // the output meta
  bbts::ud_impl_t::meta_args_t output_meta;

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

  std::vector<kernel_prep_ptr_t> retired_kernels;

  std::vector<reaper_request_ptr_t> finished_gc;

  std::vector<apply_schedule_ptr_t> apply_cmds;

  std::vector<reduce_schedule_ptr_t> reduce_cmds;
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
  void signal_reaper_done(reaper_request_ptr_t req) {

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
  }

  bool has_something() {

    std::unique_lock<std::mutex> lck;
    return (!retired_kernels.empty() || !finished_gc.empty() ||
            !apply_cmds.empty() || !reduce_cmds.empty());
  }

private:
  bool new_commands = false;

  std::vector<kernel_prep_ptr_t> retired_kernels;

  std::vector<reaper_request_ptr_t> finished_gc;

  std::vector<apply_schedule_ptr_t> apply_cmds;

  std::vector<reduce_schedule_ptr_t> reduce_cmds;

  std::vector<delete_schedule_ptr_t> delete_cmds;

  // locks the scheduler queue
  std::mutex m;

  // used for waking up and signaling
  std::condition_variable cv;
};
using scheduler_request_queue_ptr_t =
    std::shared_ptr<scheduler_request_queue_t>;

} // namespace bbts
