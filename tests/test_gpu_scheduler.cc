#include "../src/commands/command_loader.h"
#include "../src/commands/parsed_command.h"
#include "../src/storage/storage.h"
#include "../src/ud_functions/udf_manager.h"
#include "../src/utils/concurent_queue.h"
#include "../third_party/cuda/gpu.h"
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

namespace bbts {

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

class goodness_heuristic_class_t {
public:
  void tensor_loaded(tid_t id) {};

  void tensor_unloaded(tid_t id) {};

  void register_apply(const bbts::command_ptr_t &cmd) {};

  void register_reduce(const bbts::command_ptr_t &cmd) {};

  kernel_prep_ptr_t get_next() { return nullptr; };
};

class early_scheduler_t {
public:
  void tensor_loaded(tid_t id, int dev) {};

  void tensor_unloaded(tid_t id, int dev) {};

  void register_apply(const bbts::command_ptr_t &cmd) {};

  void register_reduce(const bbts::command_ptr_t &cmd) {};

  int32_t has_same_gpu() { return -1; };

  bool has_on_gpu() { return false; };

  kernel_prep_ptr_t get_next() { return nullptr; };
};

class memory_t {
public:
  
  void mark_for_use(tid_t id) {};
  void mark_for_deletion(tid_t id) {};

  void tensor_unpinned(tid_t id, int dev) {};
  void tensor_created(tid_t id, int dev) {};
  void pin(kernel_prep_ptr_t kp, int dev) {};
  int can_preallocate(kernel_prep_ptr_t kp) { return -1; };


  // go through all the GPUs
  //   can we find enough memory on the GPU? (we find the memory in the
  //   following order)
  //   - can we delete some tensors that are scheduled for deletion
  //   - can we remove some redundant tensors that are already on the GPU.
  //   - is there any unpinned memory that we can
  // if there are multiple GPUs with resources we pick the one that needs to
  // transfer the least, if they need to transfer the same we pick the most
  // free memory (or based on the workload not sure)
  void preallocate(kernel_prep_ptr_t kp, int dev) {};
  int can_gc(kernel_prep_ptr_t kp) { return -1; };
  reaper_request_ptr_t get_gc_request(kernel_prep_ptr_t kp, int dev) { return nullptr; };
};

class multi_gpu_scheduler_t {
public:
  multi_gpu_scheduler_t(size_t num_gpus, bbts::storage_ptr_t storage,
                        bbts::udf_manager_ptr udm, tensor_factory_ptr_t tf)
      : _num_gpus(num_gpus), run_queue(num_gpus), storage(std::move(storage)),
        udm(std::move(udm)), tf(std::move(tf)) {

    // p2p nvlink initialization
  }

  // runs kernels that have the tensors in memory
  void gpu_execution_thread(int32_t dev) {

    while (true) {

      // get a kernel run
      kernel_prep_ptr_t req{};
      run_queue[dev].wait_dequeue(req);

      // get the kernel
      auto kernel = req->run_me;

      // check if we are done
      if (kernel == nullptr) {
        break;
      }

      // call the kernel
      kernel->ud->call_gpu_ud(kernel->params, kernel->inputs, kernel->outputs);

      // mark that the kernels is retired now
      scheduler_queue.signal_kernel_done(req);
    }
  }

  // moves the tensors from one gpu to another, this is usually faster
  void gpu_to_gpu_thread(int32_t dev) {

    cudaStream_t cpy_stream;
    cudaStreamCreate(&cpy_stream);

    while (true) {

      // get a kernel run
      kernel_prep_ptr_t prep{};
      gpu2gpu_queue[dev].wait_dequeue(prep);

      // check if we are done
      if (prep == nullptr) {
        break;
      }

      // schedule all the copies
      for (auto [src, src_dev, input_idx, num_bytes] : prep->gpu_transfers) {
        cudaMemcpyPeerAsync(
            &prep->run_me->inputs.get_by_idx(input_idx), // destination
            dev,                                         // destination device
            src,                                         // source
            src_dev,                                     // source device
            num_bytes, // the number of bytes to copy
            cpy_stream);
      }

      // sync all the copies
      cudaStreamSynchronize(cpy_stream);

      {
        // mark that we are done and possibly shedule for execution
        std::unique_lock<std::mutex> lck(prep->m);
        prep->gpu_done = true;
        if (prep->gpu_done && prep->cpu_done) {
          run_queue[dev].enqueue(prep);
          prep->run_me = nullptr;
        }
      }
    }
  }

  void cpu_to_gpu_thread() {

    cudaStream_t cpy_stream;
    cudaStreamCreate(&cpy_stream);

    while (true) {

      // get a kernel run
      kernel_prep_ptr_t prep{};
      cpu2gpu_queue.wait_dequeue(prep);

      // check if we are done
      if (prep == nullptr) {
        break;
      }

      // schedule all the copies
      for (auto &t : prep->cpu_transfers) {

        auto src_tid = std::get<0>(t);
        auto input_idx = std::get<1>(t);
        auto num_bytes = std::get<2>(t);

        // run the local transaction move the tensor
        storage->local_transaction(
            {src_tid}, {}, [&](const storage_t::reservation_result_t &res) {
              // create the tensor
              auto ts = res.get[0].get().tensor;

              // copy the tensor from the CPU to the GPU
              cudaMemcpy(&prep->run_me->inputs.get_by_idx(input_idx), ts,
                         num_bytes, cudaMemcpyHostToDevice);
            });
      }

      {
        // mark that we are done and possibly shedule for execution
        std::unique_lock<std::mutex> lck(prep->m);
        prep->gpu_done = true;
        if (prep->gpu_done && prep->cpu_done) {
          run_queue[prep->dev].enqueue(prep);
          prep->run_me = nullptr;
        }
      }
    }
  }

  // moves memory from the CPU to GPUs
  void command_prep_thread() {

    bool should_sleep = false;
    scheduler_request_ptr_t req = std::make_shared<scheduler_request_t>();
    while (true) {

      // if we don't have anything else to do or
      if (should_sleep || scheduler_queue.has_something()) {
        scheduler_queue.wait_dequeue(req);
      }

      // 1. check for finished kernels
      for (auto &fk : req->retired_kernels) {

        // 1. unpin all the inputs
        for (auto &out : fk->input) {
          mem.tensor_unpinned(out, fk->dev);
        }

        // 2. since a new tensor has been created update commands that can be
        // scheduled
        for (auto &out : fk->output) {
          heuristic.tensor_loaded(out);
          early_scheuler.tensor_loaded(out, fk->dev);
          mem.tensor_created(out, fk->dev);
        }
      }

      // 2. did we get any new commands that were scheduled?
      // go through them and schedule them, while doing this update the
      // 'GOODNESS'
      for (auto &nc : req->apply_cmds) {

        // register the apply with the heuristic
        heuristic.register_apply(nc->cmd);
        early_scheuler.register_apply(nc->cmd);

        // mark all the inputs for use
        for(auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
          mem.mark_for_use(nc->cmd->get_inputs()[idx].tid);
        }
      }
      for (auto &nc : req->reduce_cmds) {

        // register the reduce with the heuristic
        heuristic.register_reduce(nc->cmd);
        early_scheuler.register_reduce(nc->cmd);

        // mark all the inputs for use
        for(auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
          mem.mark_for_use(nc->cmd->get_inputs()[idx].tid);
        }
      }

      // 3. check for resource freed
      for (auto &gc_req : req->finished_gc) {

        // schedule them for execution (finish reaper_request and add to
        // execution queue)
        mem.pin(gc_req->to_run, gc_req->to_run->dev);
        run_queue[gc_req->to_run->dev].enqueue(gc_req->to_run);
      }

      // 4.1. check if we have a command that we can run immeidately (all the
      // inputs are on the same GPU)
      int dev;
      if ((dev = early_scheuler.has_same_gpu()) != -1) {

        // get the command and schedule it
        kernel_prep_ptr_t kernel_prep = early_scheuler.get_next();

        // schedule them for execution (pin resources and add to execution
        // queue)
        mem.pin(kernel_prep, dev);
        run_queue[dev].enqueue(kernel_prep);
      }
      // 4.2. othwerwise check if we have commands that have inputs on at least
      // one of the GPUs
      else if (early_scheuler.has_on_gpu()) {

        // get the schedluling command
        kernel_prep_ptr_t kernel_prep = early_scheuler.get_next();

        // 4.2.1 check if we can preallocate the additional memory
        // on one of the GPUs (priority should be give to the least busy GPU)
        if ((dev = mem.can_preallocate(kernel_prep)) != -1) {

          // we can preallocate in which case we instruct the gpu2gpu threads to
          // perform the necessary copies (we don't preallocate if it is already
          // transfering just add an additional pin)
          mem.preallocate(kernel_prep, dev);

          // we only need to setup GPU2GPU transfers as all the required tensors
          // are already there
          if (!kernel_prep->gpu_transfers.empty()) {
            gpu2gpu_queue[dev].enqueue(kernel_prep);
          }
        }
        // 4.2.2 check if we can run garbage collection and then run the kernel
        else if ((dev = mem.can_gc(kernel_prep)) != -1) {

          // make a garbage collection request
          reaper_request_ptr_t gc_request =
              mem.get_gc_request(kernel_prep, dev);

          // schedule the request
          reaper_queue.enqueue(gc_request);
          should_sleep = true;
          continue;
        }
        // 4.2.3 go sleep some as there is nothing to do...
        else {

          // go and sleep a bit (unless new stuff happened in the mean time)
          should_sleep = true;
          continue;
        }
      }

      // 5.1 if there are not commands we can schedule immediately
      //  we pick a command based on a 'GOODNESS' score
      kernel_prep_ptr_t kernel_prep = heuristic.get_next();
      if (kernel_prep == nullptr) {
        continue;
      }

      /// 5.2.1 check if we can preallocate the required memory
      if ((dev = mem.can_preallocate(kernel_prep)) != -1) {

        // we can preallocate in which case we instruct the gpu2gpu threads to
        // perform the necessary copies (we don't preallocate if it is already
        // transfering just add an additional pin)
        mem.preallocate(kernel_prep, dev);

        if (!kernel_prep->cpu_transfers.empty()) {
          cpu2gpu_queue.enqueue(kernel_prep);
        }

        if (!kernel_prep->gpu_transfers.empty()) {
          gpu2gpu_queue[dev].enqueue(kernel_prep);
        }
      }
      // 5.2.2 check if we can run garbage collection and then run the kernel
      else if ((dev = mem.can_gc(kernel_prep)) != -1) {

        // make a garbage collection request
        reaper_request_ptr_t gc_request = mem.get_gc_request(kernel_prep, dev);

        // schedule the request
        reaper_queue.enqueue(gc_request);
        should_sleep = true;
        continue;
      }
    }
  }

  // reclaims memory on the GPU either by moving them back
  // to the CPU or straight up deleting them
  void reaper_thread(int dev_id) {

    cudaStream_t free_stream;
    cudaStreamCreate(&free_stream);

    while (true) {

      reaper_request_ptr_t request;
      reaper_queue.wait_dequeue(request);

      // finish if the request is null
      if (request == nullptr) {
        break;
      }

      // schedule all the free commands
      for (auto t : request->to_free) {
        cudaFreeAsync(t, free_stream);
      }

      for (auto t : request->to_evict) {

        auto mem = std::get<0>(t);
        auto tid = std::get<1>(t);
        auto num_bytes = std::get<2>(t);

        // run the local transaction move the tensor
        storage->local_transaction(
            {}, {{tid, num_bytes}},
            [&](const storage_t::reservation_result_t &res) {
              // create the tensor
              auto ts = res.create[0].get().tensor;

              // copy the tensor from the CPU to the GPU
              cudaMemcpy(mem, ts, num_bytes, cudaMemcpyDeviceToHost);
            });

        // free this memory
        cudaFree(mem);
      }

      // sync free
      cudaStreamSynchronize(free_stream);

      // signal that we have processed this request
      scheduler_queue.signal_reaper_done(request);
    }
  }

  void schedule_apply(bbts::command_ptr_t cmd) {

    assert(cmd->type == bbts::command_t::APPLY);
    auto fun = udm->get_fn_impl(cmd->fun_id);

    // the parameters
    ud_impl_t::tensor_params_t _params;
    bbts::ud_impl_t::meta_args_t input_meta;
    bbts::ud_impl_t::meta_args_t output_meta;

    // prepare the inputs
    {
      std::unique_lock<std::mutex> lck(meta_lck);

      input_meta.resize(cmd->get_inputs().size());
      for (int i = 0; i < cmd->get_inputs().size(); i++) {
        input_meta.set(i, meta[cmd->get_inputs()[i].tid]);
      }

      // prepare the outputs
      output_meta.resize(cmd->get_outputs().size());
      for (int i = 0; i < cmd->get_outputs().size(); i++) {
        output_meta.set(i, meta[cmd->get_outputs()[i].tid]);
      }
    }

    // init the parameters
    _params = ud_impl_t::tensor_params_t{._params = cmd->get_parameters()};

    // run the function to generate the metadata
    fun->get_out_meta(_params, input_meta, output_meta);

    // figure out the number of bytes
    std::vector<size_t> out_bytes(output_meta.num_args());
    for (size_t idx = 0; idx < output_meta.num_args(); ++idx) {
      out_bytes.push_back(tf->get_tensor_size(output_meta.get_by_idx(idx)));
    }
    std::vector<size_t> in_bytes(input_meta.num_args());
    for (size_t idx = 0; idx < input_meta.num_args(); ++idx) {
      in_bytes.push_back(tf->get_tensor_size(input_meta.get_by_idx(idx)));
    }

    // signal that we have processed this request
    auto apply_sch = std::make_shared<apply_schedule_t>();
    apply_sch->fn = fun;
    apply_sch->input_num_bytes = std::move(in_bytes);
    apply_sch->output_num_bytes = std::move(out_bytes);
    apply_sch->cmd = std::move(cmd);

    // signal the apply
    scheduler_queue.signal_apply(std::move(apply_sch));
  }

  void schedule_reduce(bbts::command_ptr_t cmd) {

    assert(cmd->type == bbts::command_t::REDUCE);
    auto fun = udm->get_fn_impl(cmd->fun_id);

    // the parameters
    ud_impl_t::tensor_params_t _params;
    bbts::ud_impl_t::meta_args_t input_meta;
    bbts::ud_impl_t::meta_args_t output_meta;

    // prepare the inputs
    {
      std::unique_lock<std::mutex> lck(meta_lck);

      input_meta.resize(cmd->get_inputs().size());
      for (int i = 0; i < cmd->get_inputs().size(); i++) {
        input_meta.set(i, meta[cmd->get_inputs()[i].tid]);
      }

      // prepare the outputs
      output_meta.resize(cmd->get_outputs().size());
      for (int i = 0; i < cmd->get_outputs().size(); i++) {
        output_meta.set(i, meta[cmd->get_outputs()[i].tid]);
      }
    }

    // init the parameters
    _params = ud_impl_t::tensor_params_t{._params = cmd->get_parameters()};

    // run the function to generate the metadata
    fun->get_out_meta(_params, input_meta, output_meta);

    // make the reduce request
    auto reduce_sch = std::shared_ptr<reduce_schedule_t>();

    // create the schedule request
    reduce_sch->cmd = std::move(cmd);
    reduce_sch->fn = fun;
    reduce_sch->input_meta = input_meta;
    reduce_sch->output_meta = output_meta;

    // signal the reduce
    scheduler_queue.signal_reduce(std::move(reduce_sch));
  }

  void mark_for_deletion(bbts::command_ptr_t cmd) {

    // make the reduce request
    auto delete_sch = std::shared_ptr<delete_schedule_t>();
    delete_sch->cmd = std::move(cmd);
    scheduler_queue.signal_delete(std::move(delete_sch));
  }

  void flush() {}

  void shutdown() {

    // shutdown the scheduler
    _is_running = false;
  }

  // the heuristic we use to prioritize commands
  goodness_heuristic_class_t heuristic;

  // this schedules commands that are already known to be on the GPU
  early_scheduler_t early_scheuler;

  //
  memory_t mem;

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
  concurent_queue<reaper_request_ptr_t> reaper_queue;

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