#include "gpu_profiler.h"
#include <google/protobuf/util/json_util.h>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <mutex>

bbts::gpu_profiler_t::gpu_profiler_t(size_t num_gpus) {
  for(auto dev = 0; dev < num_gpus; ++dev) {
    log.add_device_logs();
  }

  auto now = std::chrono::high_resolution_clock::now();
  base_tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

void bbts::gpu_profiler_t::log_cpu_copy_begin(bbts::tid_t id, size_t num_bytes, int32_t dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // create a new cpu2gpu transfer log
  auto &dl = log.mutable_device_logs()->at(dev);
  auto cpu2gpu = dl.add_cpu2gpu_transfer_stats();

  // fill out the data
  cpu2gpu->set_start(tick);
  cpu2gpu->set_dst_dev(dev);
  cpu2gpu->set_num_bytes(num_bytes);
}

void bbts::gpu_profiler_t::log_cpu_copy_end(int32_t dev) {
  
  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // set the end time stamp
  auto &dl = log.mutable_device_logs()->at(dev);
  dl.mutable_cpu2gpu_transfer_stats()->rbegin()->set_end(tick);
}

void bbts::gpu_profiler_t::log_gpu_copy_begin(int32_t dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // create a new cpu2gpu transfer log
  auto &dl = log.mutable_device_logs()->at(dev);
  auto cpu2gpu = dl.add_gpu2gpu_transfer_stats();

  // fill out the start values
  cpu2gpu->set_start(tick);
  cpu2gpu->set_dst_dev(dev);
}

void bbts::gpu_profiler_t::log_gpu_copy_tensor(tid_t tid, size_t num_bytes, 
                                               int32_t dst_dev, int32_t src_dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // set the end time stamp
  auto &dl = log.mutable_device_logs()->at(dst_dev);

  // set the gpu2gpu transfer
  auto t = dl.mutable_gpu2gpu_transfer_stats()->rbegin()->add_tensors();
  t->set_num_bytes(num_bytes);
  t->set_src_dev(src_dev);
  t->set_tensor(tid);
}

void bbts::gpu_profiler_t::log_gpu_copy_end(int32_t dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // set the end timestamp
  auto &dl = log.mutable_device_logs()->at(dev);
  dl.mutable_gpu2gpu_transfer_stats()->rbegin()->set_end(tick);
}

void bbts::gpu_profiler_t::log_gpu_copy_cancel(int32_t dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  auto &dl = log.mutable_device_logs()->at(dev);
  dl.mutable_gpu2gpu_transfer_stats()->RemoveLast();
}

void bbts::gpu_profiler_t::kernel_begin(const kernel_prep_ptr_t &prep) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // store the kernel
  auto &dl = log.mutable_device_logs()->at(prep->dev);
  auto ks = dl.mutable_kernels_stats()->Add();

  // store the kernel stats
  ks->set_start(tick);
  ks->set_kernel_run_idx(prep->kernel_prep_id);
}

void bbts::gpu_profiler_t::kernel_end(const kernel_prep_ptr_t &prep) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // make sure this is the same command we are ending
  auto &dl = log.mutable_device_logs()->at(prep->dev);
  // assert(dl.mutable_kernels_stats()->rbegin()->kernel_run_idx() == prep->kernel_prep_id);
  dl.mutable_kernels_stats()->rbegin()->set_end(tick);
}

void bbts::gpu_profiler_t::tensor_freed(tid_t id, int32_t dev, size_t num_bytes) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // log the tensor freed
  auto &dl = log.mutable_device_logs()->at(dev);
  auto tmp = dl.add_free_tensor_stats();
  tmp->set_start(tick);
  tmp->set_tensor(id);
  tmp->set_dst_dev(dev);
  tmp->set_num_bytes(num_bytes);
}

void bbts::gpu_profiler_t::tensor_eviction_start(tid_t id, int32_t dev, size_t num_bytes) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // set the stats
  auto &dl = log.mutable_device_logs()->at(dev);
  auto ts = dl.add_evicted_tensor_stats();
  ts->set_start(tick);
  ts->set_tensor(id);
  ts->set_dst_dev(dev);
  ts->set_num_bytes(num_bytes);
}

void bbts::gpu_profiler_t::tensor_eviction_end(tid_t id, int32_t dev) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  // set the end 
  log.mutable_device_logs()->at(dev).mutable_evicted_tensor_stats()->rbegin()->set_end(tick);
}

void bbts::gpu_profiler_t::log_kernel_scheduled(const kernel_prep_ptr_t &prp) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // get the current timestamp
  auto now = std::chrono::high_resolution_clock::now();
  auto tick = duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() - base_tick;

  auto &dl = log.mutable_device_logs()->at(prp->dev);
  auto ks = dl.add_kernels_scheduled();

  ks->set_kernel_run_id(prp->kernel_prep_id);
  ks->set_command_id(prp->command_id);
  ks->set_ud_name(prp->run_me->ud->ud_name);
  ks->set_impl_name(prp->run_me->ud->impl_name);
  ks->set_start(tick);
  ks->set_dev(prp->dev);

  for(auto idx = 0; idx < prp->input.size(); ++idx) {
    ks->add_input(prp->input[idx]);
    ks->add_input_sizes(prp->input_sizes[idx]);
  }

  for(auto idx = 0; idx < prp->output.size(); ++idx) {
    ks->add_output(prp->output[idx]);
    ks->add_output_sizes(prp->output_sizes[idx]);
  }
  
  for(auto &t : prp->cpu_transfers) {
    ks->add_cpu_transfers(t->tid);
  }

  for(auto &t : prp->gpu_transfers) {
    auto transfer = ks->add_gpu_transfers();
    transfer->set_tid(t->tid);
    transfer->set_src_dev(t->src_dev);
  }
}

void bbts::gpu_profiler_t::log_gc_scheduled(const gc_request_ptr_t &gc_req) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  auto &dl = log.mutable_device_logs()->at(gc_req->dev);
  auto ks = dl.add_gc_scheduled();
  
  ks->set_dev(gc_req->dev);
  ks->set_free_memory_used(gc_req->free_memory_used);
  ks->set_kernel_run_id(gc_req->to_run->kernel_prep_id);

  for(auto free_me : gc_req->to_free) {
    auto free = ks->add_to_free();
    free->set_num_bytes(free_me->num_bytes);
    free->set_tid(free_me->tid);
  }

  for(auto evict_me : gc_req->to_evict) {
    auto evict = ks->add_to_evict();
    evict->set_num_bytes(evict_me->num_bytes);
    evict->set_tid(evict_me->tid);
  }
}

void bbts::gpu_profiler_t::save(const std::string file_name) {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  std::ofstream ofs(file_name, std::ios_base::out | std::ios_base::binary);
  log.SerializeToOstream(&ofs);
}

std::string bbts::gpu_profiler_t::log_as_json() {

  // lock this thing
  std::unique_lock<std::mutex> lck(m);

  // convert the object to json
  std::string json_string;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  options.always_print_primitive_fields = true;
  options.preserve_proto_field_names = true;
  MessageToJsonString(log, &json_string, options);
  
  // return the json
  return std::move(json_string);
}
