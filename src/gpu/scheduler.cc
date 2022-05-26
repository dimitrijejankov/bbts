#include "scheduler.h"
#include "types.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <unistd.h>
#include <utility>

namespace bbts {

multi_gpu_scheduler_t::multi_gpu_scheduler_t(size_t num_gpus,
                                             size_t gpu_mem_size,
                                             size_t num_numa,
                                             bbts::storage_ptr_t storage,
                                             bbts::udf_manager_ptr udm,
                                             tensor_factory_ptr_t tf)
    : _num_gpus(num_gpus), run_queue(num_gpus), storage(std::move(storage)), 
      gpu2gpu_queue(num_gpus), gc_queue(num_gpus),
      udm(std::move(udm)), tf(std::move(tf)), heuristic(num_gpus), 
      mem(num_gpus, gpu_mem_size), num_unfinished_kernels(0), 
      profiler(num_gpus), cpu2gpu_queue(num_numa) {

  // set the device
  for(auto dev = 0; dev < _num_gpus; ++dev) {
    cudaSetDevice(dev);
    for (auto peer = 0; peer < _num_gpus; ++peer) {
      if (peer != dev) {
        cudaDeviceEnablePeerAccess(peer, 0);
      }
    }
  }

  // set the numa stuff
  gpus_per_num_node = num_gpus / num_numa;
}

void multi_gpu_scheduler_t::gpu_execution_thread(int32_t dev) {

  // set the device
  cudaSetDevice(dev);

  // init the stream
  cudaStream_t run_stream;
  cudaStreamCreate(&run_stream);

  // create the cublas handle
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, run_stream);

  // kick off the run process
  while (true) {

    // get a kernel run
    kernel_prep_ptr_t req{};
    run_queue[dev].wait_dequeue(req);

    // check if we are done
    if (req == nullptr) {
      break;
    }

    // start the kernel profiling
    profiler.kernel_begin(req);

    // get the kernel
    auto kernel = req->run_me;

    // set the cuda parameters
    #ifdef ENABLE_GPU
    kernel->params.stream = run_stream;
    kernel->params.cublas_handle = cublas_handle;
    #endif

    // call the kernel
    kernel->ud->call_gpu_ud(kernel->params, 
                            kernel->inputs, 
                            kernel->outputs);


    // sync the stream
    checkCudaErrors(cudaStreamSynchronize(run_stream));

    // end the kernel profiling
    profiler.kernel_end(req);

    // mark that the kernels is retired now
    scheduler_queue.signal_kernel_done(req);
  }
}

void multi_gpu_scheduler_t::gpu_to_gpu_thread(int32_t dst_dev) {
  
  checkCudaErrors(cudaSetDevice(dst_dev));

  // init the stream
  cudaStream_t cpy_stream;
  cudaStreamCreate(&cpy_stream);

  // kick off the copy process
  while (true) {

    // get a kernel run
    kernel_prep_ptr_t prep{};
    gpu2gpu_queue[dst_dev].wait_dequeue(prep);

    // check if we are done
    if (prep == nullptr) {
      break;
    }

    // make sure everything is malloced
    cudaEventSynchronize(prep->malloc_event);

    // log that we are doing the GPU2GPU copy
    profiler.log_gpu_copy_begin(dst_dev);

    // we keep all the finished transfers here
    std::vector<gpu_to_gpu_transfer_ptr_t> done_transfers; 
    done_transfers.reserve(prep->gpu_transfers.size());

    // schedule all the copies
    auto num_transfers = prep->gpu_transfers.size();
    for (int32_t idx = num_transfers - 1; idx >= 0; --idx) {

      // are there any CPU transfers that we need to wait for
      auto &t = prep->gpu_transfers[idx];

      if(t->is_finished) {
        prep->gpu_transfers.pop_back();
        continue;
      }

      if(t->depends != nullptr) {

        // try to lock access to the 
        std::unique_lock<std::mutex> lck(t->depends->m, std::defer_lock);

        // check if we can actually do the copy
        if(!lck.try_lock() || !t->depends->is_finished) {

          // we break out of the loop and wait for other transfers to be finished
          break;
        }
      }

      // log that we have have copied this tensor
      profiler.log_gpu_copy_tensor(t->tid, t->num_bytes - sizeof(tensor_t), dst_dev, t->src_dev);

      // init the copy of the tensor
      checkCudaErrors(cudaMemcpyPeerAsync(
          t->dst->get_data_ptr<void>(),    // destination
          dst_dev,                         // destination device
          t->src->get_data_ptr<void>(),    // source
          t->src_dev,                      // source device
          t->num_bytes - sizeof(tensor_t), // the number of bytes to copy
          cpy_stream));

      // copy the meta data
      t->dst->get_meta<tensor_meta_t>() = 
        t->src->get_meta<tensor_meta_t>();

      // store it as finished and remove it from the prep
      done_transfers.push_back(std::move(t));
      prep->gpu_transfers.pop_back();
    }

    // sync all the copies
    checkCudaErrors(cudaStreamSynchronize(cpy_stream));

    // log that all the copies are done
    if(!done_transfers.empty()) {
      profiler.log_gpu_copy_end(dst_dev);
    }
    else {
      profiler.log_gpu_copy_cancel(dst_dev);
    }

    // mark that it is finished
    for(auto &t : done_transfers) {
      t->is_finished = true;
    }

    // signal that all the gpu transfers are done
    scheduler_queue.signal_gpu_to_gpu_transfer_done(done_transfers);

    // did we manage to do all the transfers
    if(!prep->gpu_transfers.empty()) {

      // if we did not manage to process all copy requests reschedule it
      gpu2gpu_queue[dst_dev].enqueue_copy(prep);
      continue;
    }

    // mark that we are done and possibly shedule for execution
    std::unique_lock<std::mutex> lck(prep->m);
    prep->gpu_done = true;
    if (prep->gpu_done && prep->cpu_done) {
      run_queue[dst_dev].enqueue_copy(prep);
    }
  }
}

void multi_gpu_scheduler_t::cpu_to_gpu_thread(int32_t numa_node) {

  cudaStream_t cpy_stream;
  cudaStreamCreate(&cpy_stream);

  std::vector<cpu_to_gpu_transfer_ptr_t> done_transfers;
  while (true) {

    // get a kernel run
    kernel_prep_ptr_t prep{};
    cpu2gpu_queue[numa_node].wait_dequeue(prep);

    // check if we are done
    if (prep == nullptr) {
      break;
    }

    // make sure everything is malloced
    checkCudaErrors(cudaEventSynchronize(prep->malloc_event));
    checkCudaErrors(cudaSetDevice(prep->dev));

    // schedule all the copies
    done_transfers.clear();
    auto num_transfers = prep->cpu_transfers.size();
    for (int32_t idx = num_transfers - 1; idx >= 0; --idx) {

      // grab the transfer
      auto t = prep->cpu_transfers[idx];
      
      // convert the gpu tid to cpu tid, for a regular tensor 
      // they are the same, for an anonymous tensor they differ
      auto cpu_tid = t->tid;
      if(t->tid < 0){
        std::unique_lock<std::mutex> lck(anon_lck);
        cpu_tid = _anon_gpu_cpu[t->tid];
      }

      // lock so that we can perform the copy
      std::unique_lock<std::mutex> lck(t->m);

      // check if there is 
      if(t->depends != nullptr) {

        // try to lock access to the 
        std::unique_lock<std::mutex> lck(t->depends->m, std::defer_lock);

        // check if we can actually do the copy
        if(!lck.try_lock() || !t->depends->evicted) {

          // we break out of the loop and wait for other transfers to be finished
          break;
        }
      }

      // run the local transaction move the tensor
      if(!t->is_finished) {

        // log that we are starting a CPU2GPU copy
        profiler.log_cpu_copy_begin(t->tid, t->num_bytes - sizeof(tensor_t), t->dst_dev);

        // process the local transaction
        storage->local_transaction(
            {cpu_tid}, {}, [&](const storage_t::reservation_result_t &res) {

              // create the tensor
              auto ts = res.get[0].get().tensor;

              // copy the tensor from the CPU to the GPU
              auto dst = t->dst->get_data_ptr<void>();
              auto src = ts->get_data_ptr<void>();

              checkCudaErrors(cudaMemcpyAsync(dst, src, 
                                              t->num_bytes - sizeof(tensor_t), 
                                              cudaMemcpyHostToDevice, cpy_stream));
              checkCudaErrors(cudaStreamSynchronize(cpy_stream));

              // copy the meta data
              t->dst->get_meta<tensor_meta_t>() = 
                ts->get_meta<tensor_meta_t>();
        });
        
        // load the we have finished the copy
        profiler.log_cpu_copy_end(t->dst_dev);

        // mark that it is finished
        t->is_finished = true;
        done_transfers.push_back(std::move(t));
      }
          
      // store it as finished and remove it from the prep
      prep->cpu_transfers.pop_back();
    }

    // signal that the CPU transfers are done
    scheduler_queue.signal_cpu_to_gpu_transfer_done(done_transfers);

    // did we manage to do all the transfers
    if(!prep->cpu_transfers.empty()) {

      // if we did not manage to process all copy requests reschedule it
      cpu2gpu_queue[numa_node].enqueue_copy(prep);
      continue;
    }

    // mark that we are done and possibly shedule for execution
    std::unique_lock<std::mutex> lck(prep->m);
    prep->cpu_done = true;
    if (prep->gpu_done && prep->cpu_done) {
      run_queue[prep->dev].enqueue_copy(prep);
    }
  }
}

void multi_gpu_scheduler_t::command_prep_thread() {

  bool should_sleep = false;
  std::vector<tid_t> deleted_tensors;
  scheduler_request_ptr_t req = std::make_shared<scheduler_request_t>();
  while (true) {

    // check if we had a flush request in the mean time
    // if we don't have anything to run, just kick off the requested flush
    if(!outstanding_flush_requests.empty() && 
       !scheduler_queue.has_something() && 
       !heuristic.has_something() &&
       num_unfinished_kernels == 0) {
      
      // perform the flush this should always succeed as have nothing to do
      _perform_flush();
    }

    // if we don't have anything else to do 
    // or the scheduler has something that was 
    req->clear();
    if (should_sleep || scheduler_queue.has_something()) {
      scheduler_queue.wait_dequeue(req);
    }

    // check if we got a shutdown request if we did propagate the shutdown
    if(req->shutdown) {

      // finish all
      _perform_shutdown();
      break;
    }

    // 0. mark all the tensors that became available on the CPU
    for(auto &t : req->cpu_created_tensors) {
      auto [tid, num_bytes] = t;
      mem.tensor_loaded_on_cpu(tid, num_bytes);
      heuristic.tensor_on_cpu(tid); 
    }

    // 1. check for flush requests
    for (auto &fr : req->flush_requests) {
      outstanding_flush_requests.push_back(std::move(fr));
    }

    // 3. check for finished transfers
    for(auto &gpu_transfer : req->gpu_transfers) {
      mem.mark_transfer_done(gpu_transfer);
    }

    for(auto &cpu_transfer : req->cpu_transfers) {
      mem.mark_transfer_done(cpu_transfer);
    }

    // 2. check for finished kernels
    for (auto &fk : req->retired_kernels) {

      // 2.1. since a new tensor has been created update commands that can be scheduled
      deleted_tensors.clear();
      mem.finish_kernel_prep(fk, deleted_tensors);
      for (auto out_idx = 0; out_idx < fk->output.size(); ++out_idx) {
        heuristic.tensor_loaded(fk->output[out_idx], fk->dev);
      }

      // 2.2. we finished a kernel so decrement this
      num_unfinished_kernels--;

      // 2.3 mark that we have deleted the tensors
      for(auto t : deleted_tensors) {
        if(t < 0) {
          std::unique_lock<std::mutex> anon_locked(anon_lck);
          auto it = _anon_cpu_gpu.find(t);
          if(it != _anon_cpu_gpu.end()) {
            _deleted_tensors.enqueue_copy(it->second);
          }
        }
        else {
          _deleted_tensors.enqueue_copy(t);
        }        
      }
    }

    // 4. did we get any new commands that were scheduled?
    // go through them and schedule them, while doing this update the 'GOODNESS'
    for (auto &nc : req->to_schedule) {
      
      if(nc->cmd->type == command_t::APPLY) {

        // mark all the inputs for use
        mem.mark_for_use(nc);

        // register the apply with the heuristic
        heuristic.register_apply(nc);
      }
      else if (nc->cmd->type == command_t::REDUCE) {

        // mark all the inputs for use
        mem.mark_for_use(nc);

        // register the reduce with the heuristic
        heuristic.register_reduce(nc);
      }
      else if (nc->cmd->type == command_t::DELETE) {

        // mark all the tensor for deletion
        for(auto idx = 0; idx < nc->cmd->get_num_inputs(); ++idx) {
          auto t = nc->cmd->get_inputs()[idx].tid;
          if(mem.mark_for_deletion(nc->cmd->get_inputs()[idx].tid)) {
            if(t < 0) {
              std::unique_lock<std::mutex> anon_locked(anon_lck);
              auto it = _anon_cpu_gpu.find(t);
              if(it != _anon_cpu_gpu.end()) {
                _deleted_tensors.enqueue_copy(it->second);
              }
            }
            else {
              _deleted_tensors.enqueue_copy(t);
            }  
          }
          heuristic.remove_tensor(nc->cmd->get_inputs()[idx].tid);
        }
      }
    }

    // 5. check for resource freed
    for (auto &gc_req : req->finished_gc) {

      // schedule them for execution (finish reaper_request and add to execution queue)
      mem.finish_gc_request(gc_req);
      mem.preallocate(gc_req->to_run, gc_req->dev);

      // schedule the CPU transfers
      gc_req->to_run->cpu_done = gc_req->to_run->cpu_transfers.empty();
      if (!gc_req->to_run->cpu_transfers.empty()) {
        auto numa = get_numa_idx(gc_req->to_run->dev);
        cpu2gpu_queue[numa].enqueue_copy(gc_req->to_run);
      }

      // schedule the GPU transfers
      gc_req->to_run->gpu_done = gc_req->to_run->gpu_transfers.empty();
      if (!gc_req->to_run->gpu_transfers.empty()) {
        gpu2gpu_queue[gc_req->dev].enqueue_copy(gc_req->to_run);
      }

      // if there are not transfers to be scheduled we can just run it immediately 
      if(gc_req->to_run->cpu_transfers.empty() &&
         gc_req->to_run->gpu_transfers.empty()) {
        run_queue[gc_req->dev].enqueue_copy(gc_req->to_run);
      }

      // log that the kernel scheduled
      profiler.log_kernel_scheduled(gc_req->to_run);
    }

    // 6.1. check if we have a command that we can run immeidately (all the
    // inputs are on the same GPU) ex. APPLY 1 2 3 -> 4 | GPU 0 has 1 2 3
    auto [kernel_prep, dev] = heuristic.get_next_on_same(preffered_dev);
    if (dev != -1) {

      // schedule it for execution, if it fails to schedule put it to sleep
      bool scheduled = _schedule_for_execution(kernel_prep, dev);
      if(scheduled) {
        heuristic.mark_as_scheduled(kernel_prep);
      }

      // go to sleep if we did not manage to schedule anything
      should_sleep = !scheduled;
      continue;
    }
    // 6.2. othwerwise check if we have commands that have inputs on at least
    // one of the GPUs ex. APPLY 1 2 3 -> 4 | GPU 0 has 1 2 | GPU 1 has 3
    else if ((kernel_prep = heuristic.get_next_on_any()) != nullptr) {

      // schedule them for execution, if it fails to schedule put it to sleep
      bool scheduled = _schedule_for_execution(kernel_prep, preffered_dev);
      if(scheduled) {
        heuristic.mark_as_scheduled(kernel_prep);
      }

      // go to sleep if we did not manage to schedule anything
      should_sleep = !scheduled;
      continue;
    }

    // 7.1. if there are not commands we can schedule immediately
    //  we pick a command based on a 'GOODNESS' score
    kernel_prep = heuristic.get_next_heuristic();
    if (kernel_prep == nullptr) {
      should_sleep = true;
      continue;
    }

    // 7.2. schedule it for execution, if it fails to schedule put it to sleep
    bool scheduled = _schedule_for_execution(kernel_prep, preffered_dev);
    if(scheduled) {
      heuristic.mark_as_scheduled(kernel_prep);
    }

    // go to sleep if we did not manage to schedule anything
    should_sleep = !scheduled;
    continue;
  }
}

void multi_gpu_scheduler_t::gc_thread(int dev_id) {


  checkCudaErrors(cudaSetDevice(dev_id));

  cudaStream_t free_stream;
  cudaStreamCreate(&free_stream);

  cudaStream_t copy_stream;
  cudaStreamCreate(&copy_stream);

  while (true) {

    gc_request_ptr_t request;
    gc_queue[dev_id].wait_dequeue(request);

    // finish if the request is null
    if (request == nullptr) {
      break;
    }

    // schedule all the free commands
    for (auto &t : request->to_free) {
      auto mem = t->tensor->get_data_ptr<void>();
      checkCudaErrors(cudaFreeAsync(mem, free_stream));
      profiler.tensor_freed(t->tid, dev_id, t->num_bytes);
    }

    // evict everything we need to
    for (auto &t : request->to_evict) {

      // if this is a regular tensor we are fine
      auto cpu_tid = t->tid;
      bool has_tensor = cpu_tid > 0;

      // check if we already have an id for this tensor
      if(!has_tensor) {
        std::unique_lock<std::mutex> lck(anon_lck);
        auto it = _anon_cpu_gpu.find(cpu_tid);
        if(it != _anon_cpu_gpu.end()) {
          has_tensor = true;
          cpu_tid = it->second;
        }
      }

      // check if this is a regular tensor or an anonymous tensor
      if(has_tensor) {

        // run the local transaction move the tensor
        profiler.tensor_eviction_start(t->tid, dev_id, t->num_bytes - sizeof(tensor_t));
        storage->local_transaction({}, {{cpu_tid, t->num_bytes}},
          [&](const storage_t::reservation_result_t &res) {

            // create the tensor
            auto ts = res.create[0].get().tensor;

            // copy the tensor from the CPU to the GPU
            checkCudaErrors(cudaMemcpyAsync(ts->get_data_ptr<void>(), 
                      t->tensor->get_data_ptr<void>(), 
                      t->num_bytes - sizeof(tensor_t), 
                      cudaMemcpyDeviceToHost, copy_stream));
            checkCudaErrors(cudaStreamSynchronize(copy_stream));

            // copy the meta
            ts->get_meta<tensor_meta_t>() = 
              t->tensor->get_meta<tensor_meta_t>();
        });
        profiler.tensor_eviction_end(t->tid, dev_id);
      }
      else {

        // run the local transaction move the tensor
        profiler.tensor_eviction_start(t->tid, dev_id, t->num_bytes - sizeof(tensor_t));
        storage->local_transaction({}, {{cpu_tid, t->num_bytes}},
          [&](const storage_t::reservation_result_t &res) {

            // create the tensor
            auto ts = res.create[0].get().tensor;
            cpu_tid = res.create[0].get().id;

            // copy the tensor from the CPU to the GPU
            checkCudaErrors(cudaMemcpyAsync(ts->get_data_ptr<void>(), 
                      t->tensor->get_data_ptr<void>(), 
                      t->num_bytes - sizeof(tensor_t), 
                      cudaMemcpyDeviceToHost, copy_stream));
            checkCudaErrors(cudaStreamSynchronize(copy_stream));

            // copy the meta
            ts->get_meta<tensor_meta_t>() = 
              t->tensor->get_meta<tensor_meta_t>();
        });
        profiler.tensor_eviction_end(t->tid, dev_id);

        // store the anonymous tid mapping
        {
          std::unique_lock<std::mutex> lck(anon_lck);
          _anon_gpu_cpu[t->tid] = cpu_tid;
          _anon_cpu_gpu[cpu_tid] = t->tid;
        }
      }

      // mark that the tensor now was evicted
      {
        std::unique_lock<std::mutex> lck(t->m);
        t->evicted = true;
      }

      // free this memory
      assert(t->tensor->get_data_ptr<void>() != nullptr);
      checkCudaErrors(cudaFree(t->tensor->get_data_ptr<void>()));
    }

    // sync free
    checkCudaErrors(cudaStreamSynchronize(free_stream));

    // signal that we have processed this request
    scheduler_queue.signal_reaper_done(request);
  }
}

std::vector<tid_t> multi_gpu_scheduler_t::get_deleted_tensors() {

  std::vector<tid_t> deleted_tensors;
  if(!_deleted_tensors.wait_dequeue_all(deleted_tensors)) {
    return {};
  }
  return std::move(deleted_tensors);
}

void multi_gpu_scheduler_t::save_log(const std::string file_name) {
  profiler.save(file_name);
}

size_t multi_gpu_scheduler_t::get_num_numa() const {
  return _num_gpus / gpus_per_num_node;
}

int32_t multi_gpu_scheduler_t::get_numa_idx(int32_t gpu_idx) const {
  return gpu_idx / gpus_per_num_node;
}

gpu_command_schedule_ptr_t multi_gpu_scheduler_t::_prepare_apply(bbts::command_ptr_t &cmd) {

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
      input_meta.set(i, _meta[cmd->get_inputs()[i].tid]);
    }

    // prepare the outputs
    output_meta.resize(cmd->get_outputs().size());
    for (int i = 0; i < cmd->get_outputs().size(); i++) {
      output_meta.set(i, _meta[cmd->get_outputs()[i].tid]);
    }
  }

  // init the parameters
  _params = ud_impl_t::tensor_params_t{._params = cmd->get_parameters()};

  // run the function to generate the metadata
  fun->get_out_meta(_params, input_meta, output_meta);

  // figure out the number of bytes
  std::vector<size_t> out_bytes(output_meta.num_args());
  for (size_t idx = 0; idx < output_meta.num_args(); ++idx) {
    out_bytes[idx] = tf->get_tensor_size(output_meta.get_by_idx(idx));
  }
  std::vector<size_t> in_bytes(input_meta.num_args());
  for (size_t idx = 0; idx < input_meta.num_args(); ++idx) {
    in_bytes[idx] = tf->get_tensor_size(input_meta.get_by_idx(idx));
  }

  // signal that we have processed this request
  auto apply_sch = std::make_shared<gpu_command_schedule_t>();
  apply_sch->fn = fun;
  apply_sch->input_sizes = std::move(in_bytes);
  apply_sch->output_sizes = std::move(out_bytes);
  apply_sch->params = _params;
  apply_sch->cmd = std::move(cmd);

  // signal the apply
  return std::move(apply_sch);
}

gpu_command_schedule_ptr_t multi_gpu_scheduler_t::_prepare_reduce(bbts::command_ptr_t &cmd) {

  assert(cmd->type == bbts::command_t::REDUCE);
  auto fun = udm->get_fn_impl(cmd->fun_id);

  // the parameters
  ud_impl_t::tensor_params_t _params;
  bbts::ud_impl_t::meta_args_t input_meta_for_size;
  bbts::ud_impl_t::meta_args_t input_meta;
  bbts::ud_impl_t::meta_args_t output_meta;

  // TODO this is effectively cheating and we assume that the size 
  // of each intermediate result is the same... it is not
  {
    std::unique_lock<std::mutex> lck(meta_lck);

    // prepare the inputs for the kernel meta run
    input_meta_for_size.resize(2);
    for (int i = 0; i < 2; i++) {
      input_meta_for_size.set(i, _meta[cmd->get_inputs()[i].tid]);
    }

    // perpare the inputs for the scheduler
    input_meta.resize(cmd->get_num_inputs());
    for (int i = 0; i < cmd->get_num_inputs(); i++) {
      input_meta.set(i, _meta[cmd->get_inputs()[i].tid]);
    }

    // prepare the outputs
    output_meta.resize(1);
    output_meta.set(0, _meta[cmd->get_outputs()[0].tid]);
  }

  // init the parameters
  _params = ud_impl_t::tensor_params_t{._params = cmd->get_parameters()};

  // run the function to generate the metadata
  fun->get_out_meta(_params, input_meta_for_size, output_meta);

  // make the reduce request
  auto reduce_sch = std::make_shared<gpu_command_schedule_t>();

  // create the schedule request
  reduce_sch->cmd = std::move(cmd);
  reduce_sch->fn = fun;
  for (int i = 0; i < input_meta.num_args(); i++) {
    reduce_sch->input_sizes.push_back(tf->get_tensor_size(input_meta.get_by_idx(i)));
  }
  reduce_sch->output_sizes = { tf->get_tensor_size(output_meta.get_by_idx(0)) };

  // signal the reduce
  return std::move(reduce_sch);
}

gpu_command_schedule_ptr_t multi_gpu_scheduler_t::_prepare_delete(bbts::command_ptr_t &cmd) {

  // make the reduce request
  auto delete_sch = std::make_shared<gpu_command_schedule_t>();
  delete_sch->cmd = std::move(cmd);
  return std::move(delete_sch);
}

void multi_gpu_scheduler_t::mark_tensor_on_cpu(tid_t tid, 
                                               size_t num_bytes, 
                                               tensor_meta_t meta) {
  scheduler_queue.signal_tensor_on_cpu(tid, num_bytes);
  std::unique_lock<std::mutex> lck(meta_lck);
  _meta[tid] = meta;
}

void multi_gpu_scheduler_t::flush() { 
  auto success =  scheduler_queue.signal_flush_request().get();
  assert(success);
}

void multi_gpu_scheduler_t::_perform_flush() {

  std::vector<std::tuple<tensor_t*, tid_t, size_t, int32_t>> to_flush;
  mem.get_tensors_to_flush(to_flush);
  bool success = to_flush.empty();
  if(!to_flush.empty()) {

    // flush them all
    for(auto t : to_flush) {
      
      // get the info about the tensor
      auto flush_me = std::get<0>(t);
      auto tid = std::get<1>(t);
      auto num_bytes = std::get<2>(t);
      auto dev = std::get<3>(t);

      // run the local transaction move the tensor
      storage->local_transaction(
          {}, {{tid, num_bytes}},
          [&](const storage_t::reservation_result_t &res) {
            
            // set the device
            cudaSetDevice(dev);

            // create the tensor
            auto ts = res.create[0].get().tensor;

            // copy the tensor from the CPU to the GPU
            checkCudaErrors(cudaMemcpy(ts->get_data_ptr<void>(), 
                       flush_me->get_data_ptr<void>(), 
                       num_bytes - sizeof(tensor_t), 
                       cudaMemcpyDeviceToHost));

            // set the meta data
            ts->get_meta<bbts::tensor_meta_t>() = _meta[tid];
          });
    }
    success = true;
  }
  mem.mark_as_flushed(to_flush);

  // update all the outstanding flushs requests
  for(auto &fr : outstanding_flush_requests) {
    fr->status.set_value(success);
  }
  outstanding_flush_requests.clear();
}

void multi_gpu_scheduler_t::_perform_shutdown() {

  // flush everything with null pointers
  auto kp_done = kernel_prep_ptr_t(nullptr);
  auto gc_done = gc_request_ptr_t(nullptr);
  for(auto dev = 0; dev < _num_gpus; ++dev) {
    gpu2gpu_queue[dev].enqueue_copy(kp_done);
    run_queue[dev].enqueue_copy(kp_done);
    gc_queue[dev].enqueue_copy(gc_done);
  }
  
  for(auto numa = 0; numa < get_num_numa(); numa++) {
    cpu2gpu_queue[numa].enqueue_copy(kp_done);
  }

  _deleted_tensors.shutdown();
}

void multi_gpu_scheduler_t::shutdown() {

  // shutdown the scheduler
  scheduler_queue.signal_shutdown();
}


int32_t multi_gpu_scheduler_t::num_gpus() const {
  return _num_gpus;
}

void multi_gpu_scheduler_t::schedule(std::vector<bbts::command_ptr_t> &to_schedule) {

  std::vector<bbts::gpu_command_schedule_ptr_t> prep;
  for(auto &cmd : to_schedule) {
    
    assert(cmd->is_apply() || cmd->is_reduce() || cmd->is_delete());
    if(cmd->is_apply()) {
      prep.push_back(_prepare_apply(cmd));
    }
    else if(cmd->is_reduce()) {
      prep.push_back(_prepare_reduce(cmd));
    }
    else if(cmd->is_delete()) {
      prep.push_back(_prepare_delete(cmd));
    }
  }
  scheduler_queue.signal_cmds_scheduled(prep);
}


bool multi_gpu_scheduler_t::_schedule_for_execution(kernel_prep_ptr_t kernel_prep, int32_t target_dev) {
  
  int32_t dev;
  if ((dev = mem.can_preallocate(kernel_prep, target_dev)) != -1) {

    // we can preallocate in which case we instruct the gpu2gpu threads to
    // perform the necessary copies (we don't preallocate if it is already
    // transfering just add an additional pin)
    mem.preallocate(kernel_prep, dev);

    // we only need to setup GPU2GPU transfers as all the required tensors
    // are already there
    kernel_prep->gpu_done = kernel_prep->gpu_transfers.empty();
    if (!kernel_prep->gpu_transfers.empty()) {
      gpu2gpu_queue[dev].enqueue_copy(kernel_prep);
    }

    // schedule the CPU transfers
    kernel_prep->cpu_done = kernel_prep->cpu_transfers.empty();
    if (!kernel_prep->cpu_transfers.empty()) {
      auto numa = get_numa_idx(kernel_prep->dev);
      cpu2gpu_queue[numa].enqueue_copy(kernel_prep);
    }

    // if there are not transfers to be scheduled we can just run it immediately 
    if(kernel_prep->cpu_transfers.empty() &&
       kernel_prep->gpu_transfers.empty()) {
      run_queue[dev].enqueue_copy(kernel_prep);
    }

    // log that the kernel scheduled
    profiler.log_kernel_scheduled(kernel_prep);

    // we just scheduled a kernel
    num_unfinished_kernels++;
    
    // go to the next device
    preffered_dev = (preffered_dev + 1) % _num_gpus;

    return true;
  }
  // check if we can run garbage collection and then run the kernel
  else if ((dev = mem.can_gc(kernel_prep, target_dev)) != -1) {

    // make a garbage collection request
    gc_request_ptr_t gc_request = mem.get_gc_request(kernel_prep, dev);
    
    // log that the garbage collection is scheduled
    profiler.log_gc_scheduled(gc_request);

    // schedule the request
    gc_queue[dev].enqueue_copy(gc_request);

    // we jusst schduled a kernel
    num_unfinished_kernels++;
    
    // go to the next device
    preffered_dev = (preffered_dev + 1) % _num_gpus;

    return true;
  }
  
  return false;
}

} // namespace bbts