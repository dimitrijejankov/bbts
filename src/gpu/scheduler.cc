#include "scheduler.h"
#include <cstddef>
#include <mutex>

namespace bbts {

multi_gpu_scheduler_t::multi_gpu_scheduler_t(size_t num_gpus,
                                             size_t gpu_mem_size,
                                             bbts::storage_ptr_t storage,
                                             bbts::udf_manager_ptr udm,
                                             tensor_factory_ptr_t tf)
    : _num_gpus(num_gpus), run_queue(num_gpus), storage(std::move(storage)),
      udm(std::move(udm)), tf(std::move(tf)), heuristic(num_gpus), mem(num_gpus, gpu_mem_size) {

  // TODO p2p nvlink initialization
}
void multi_gpu_scheduler_t::gpu_execution_thread(int32_t dev) {

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

void multi_gpu_scheduler_t::gpu_to_gpu_thread(int32_t dev) {

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

    // we keep all the finished transfers here
    std::vector<gpu_to_gpu_transfer_ptr_t> done_transfers; 
    done_transfers.reserve(prep->gpu_transfers.size());

    // schedule all the copies
    auto num_transfers = prep->gpu_transfers.size();
    for (auto idx = num_transfers - 1; idx >= 0; --idx) {

      // are there any CPU transfers that we need to wait for
      auto &t = prep->gpu_transfers[idx];
      if(t->depends != nullptr) {

        // try to lock access to the 
        std::unique_lock<std::mutex> lck(t->depends->m, std::defer_lock);

        // check if we can actually do the copy
        if(!lck.try_lock() || !t->depends->is_finished) {

          // we break out of the loop and wait for other transfers to be finished
          break;
        }
      }

      // init the copy of the tensor
      cudaMemcpyPeerAsync(
          t->dst,             // destination
          dev,                // destination device
          t->src,             // source
          t->src_dev,         // source device
          t->num_bytes,       // the number of bytes to copy
          cpy_stream);

      // store it as finished and remove it from the prep
      done_transfers.push_back(std::move(t));
      prep->gpu_transfers.pop_back();
    }

    // sync all the copies
    cudaStreamSynchronize(cpy_stream);

    // signal that all the gpu transfers are done
    scheduler_queue.signal_gpu_to_gpu_transfer_done(done_transfers);
    
    // did we manage to do all the transfers
    if(!prep->gpu_transfers.empty()) {

      // if we did not manage to process all copy requests reschedule it
      gpu2gpu_queue[dev].enqueue(prep);
      continue;
    }

    // mark that we are done and possibly shedule for execution
    std::unique_lock<std::mutex> lck(prep->m);
    prep->gpu_done = true;
    if (prep->gpu_done && prep->cpu_done) {
      run_queue[dev].enqueue(prep);
    }
  }
}

void multi_gpu_scheduler_t::cpu_to_gpu_thread() {

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

      // lock so that we can perform the copy
      std::unique_lock<std::mutex> lck(t->m);

      // run the local transaction move the tensor
      storage->local_transaction(
          {t->tid}, {}, [&](const storage_t::reservation_result_t &res) {

            // create the tensor
            auto ts = res.get[0].get().tensor;

            // copy the tensor from the CPU to the GPU
            cudaMemcpy(t->dst, ts, t->num_bytes, cudaMemcpyHostToDevice);
          });

      // mark that it is finished
      t->is_finished = true;
    }

    // signal that the CPU transfers are done
    scheduler_queue.signal_cpu_to_gpu_transfer_done(prep->cpu_transfers);

    // mark that we are done and possibly shedule for execution
    std::unique_lock<std::mutex> lck(prep->m);
    prep->gpu_done = true;
    if (prep->gpu_done && prep->cpu_done) {
      run_queue[prep->dev].enqueue(prep);
    }
  }
}

void multi_gpu_scheduler_t::command_prep_thread() {

  int preffered_dev = 0;
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
      mem.unpin_all(fk, fk->dev);

      // 2. since a new tensor has been created update commands that can be
      // scheduled
      for (auto &out : fk->output) {
        heuristic.tensor_loaded(out, fk->dev);
        mem.tensor_created(out, fk->dev);
      }
    }

    // 2. check for finished transfers
    for(auto &gpu_transfer : req->gpu_transfers) {
      mem.mark_transfer_done(gpu_transfer);
    }

    for(auto &cpu_transfers : req->cpu_transfers) {
      mem.mark_transfer_done(cpu_transfers);
    }

    // 3. did we get any new commands that were scheduled?
    // go through them and schedule them, while doing this update the
    // 'GOODNESS'
    for (auto &nc : req->apply_cmds) {

      // mark all the inputs for use
      for (auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
        mem.mark_for_use(nc->cmd->get_inputs()[idx].tid);
      }

      // register the apply with the heuristic
      heuristic.register_apply(nc);
    }
    for (auto &nc : req->reduce_cmds) {

      // mark all the inputs for use
      for (auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
        mem.mark_for_use(nc->cmd->get_inputs()[idx].tid);
      }

      // register the reduce with the heuristic
      heuristic.register_reduce(nc);
    }

    // 4. check for resource freed
    for (auto &gc_req : req->finished_gc) {

      // schedule them for execution (finish reaper_request and add to
      // execution queue)
      mem.pin_all(gc_req->to_run, gc_req->to_run->dev);

      /// TODO Add if necessary to the required CPU2GPU thread and GPU2CPU
      /// thread
      //  if not not necessary just move it to the run thread
    }

    // 5.1. check if we have a command that we can run immeidately (all the
    // inputs are on the same GPU) ex. APPLY 1 2 3 -> 4 | GPU 0 has 1 2 3
    auto [kernel_prep, dev] = heuristic.get_next_on_same(preffered_dev);
    if (dev != -1) {

      // schedule them for execution (pin resources and add to execution
      // queue)
      mem.pin_all(kernel_prep, dev);
      run_queue[dev].enqueue(kernel_prep);
    }
    // 5.2. othwerwise check if we have commands that have inputs on at least
    // one of the GPUs ex. APPLY 1 2 3 -> 4 | GPU 0 has 1 2 | GPU 1 has 3
    else if ((kernel_prep = heuristic.get_next_on_any()) == nullptr) {

      // 5.2.1 check if we can preallocate the additional memory
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
      // 5.2.2 check if we can run garbage collection and then run the kernel
      else if ((dev = mem.can_gc(kernel_prep)) != -1) {

        // make a garbage collection request
        gc_request_ptr_t gc_request = mem.get_gc_request(kernel_prep, dev);

        // schedule the request
        gc_queue.enqueue(gc_request);
        should_sleep = true;
        continue;
      }
      // 5.2.3 go sleep some as there is nothing to do...
      else {

        // go and sleep a bit (unless new stuff happened in the mean time)
        should_sleep = true;
        continue;
      }
    }

    // 6.1 if there are not commands we can schedule immediately
    //  we pick a command based on a 'GOODNESS' score
    kernel_prep = heuristic.get_next_heuristic();
    if (kernel_prep == nullptr) {
      should_sleep = true;
      continue;
    }

    /// 7.2.1 check if we can preallocate the required memory
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
    // 7.2.2 check if we can run garbage collection and then run the kernel
    else if ((dev = mem.can_gc(kernel_prep)) != -1) {

      // make a garbage collection request
      gc_request_ptr_t gc_request = mem.get_gc_request(kernel_prep, dev);

      // schedule the request
      gc_queue.enqueue(gc_request);
      should_sleep = true;
      continue;
    }
  }
}
void multi_gpu_scheduler_t::gc_thread(int dev_id) {

  cudaStream_t free_stream;
  cudaStreamCreate(&free_stream);

  while (true) {

    gc_request_ptr_t request;
    gc_queue.wait_dequeue(request);

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
void multi_gpu_scheduler_t::schedule_apply(bbts::command_ptr_t cmd) {

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
  apply_sch->input_sizes = std::move(in_bytes);
  apply_sch->output_sizes = std::move(out_bytes);
  apply_sch->cmd = std::move(cmd);

  // signal the apply
  scheduler_queue.signal_apply(std::move(apply_sch));
}

void multi_gpu_scheduler_t::schedule_reduce(bbts::command_ptr_t cmd) {

  assert(cmd->type == bbts::command_t::REDUCE);
  auto fun = udm->get_fn_impl(cmd->fun_id);

  // the parameters
  ud_impl_t::tensor_params_t _params;
  bbts::ud_impl_t::meta_args_t input_meta;
  bbts::ud_impl_t::meta_args_t output_meta;

  // TODO this is effectively cheating and we assume that the size 
  // of each intermediate result is the same... it is not
  {
    std::unique_lock<std::mutex> lck(meta_lck);

    // prepare the inputs
    input_meta.resize(2);
    for (int i = 0; i < 2; i++) {
      input_meta.set(i, meta[cmd->get_inputs()[i].tid]);
    }

    // prepare the outputs
    output_meta.resize(1);
    output_meta.set(0, meta[cmd->get_outputs()[0].tid]);
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
  reduce_sch->input_sizes = { tf->get_tensor_size(input_meta.get_by_idx(0)),
                              tf->get_tensor_size(input_meta.get_by_idx(1)) };
  reduce_sch->output_size = { tf->get_tensor_size(output_meta.get_by_idx(0)) };

  // signal the reduce
  scheduler_queue.signal_reduce(std::move(reduce_sch));
}
void multi_gpu_scheduler_t::mark_for_deletion(bbts::command_ptr_t cmd) {

  // make the reduce request
  auto delete_sch = std::shared_ptr<delete_schedule_t>();
  delete_sch->cmd = std::move(cmd);
  scheduler_queue.signal_delete(std::move(delete_sch));
}
void multi_gpu_scheduler_t::flush() {}
void multi_gpu_scheduler_t::shutdown() {

  // shutdown the scheduler
  _is_running = false;
}
} // namespace bbts