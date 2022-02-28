#pragma once

#include "types.h"
#include "early_scheduler.h"
#include "goodness_heuristic.h"
#include "gpu_memory.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"

namespace bbts {

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
        for (auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
          mem.mark_for_use(nc->cmd->get_inputs()[idx].tid);
        }
      }
      for (auto &nc : req->reduce_cmds) {

        // register the reduce with the heuristic
        heuristic.register_reduce(nc->cmd);
        early_scheuler.register_reduce(nc->cmd);

        // mark all the inputs for use
        for (auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
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