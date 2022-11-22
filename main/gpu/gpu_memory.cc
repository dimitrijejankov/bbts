#include "gpu_memory.h"
#include "gpu_profiler.h"
#include "types.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_set>

#ifdef ENABLE_GPU
#include <cuda.h>
#endif

#include <memory>
#include <ostream>
#include <vector>

namespace bbts {

gpu_memory_t::gpu_memory_t(size_t num_devices, size_t mem_per_gpu)
    : _num_devices(num_devices) {

  _mem_per_gpu = mem_per_gpu;
  _unpinned_tensors.resize(num_devices);
  _total_free.resize(num_devices);
  _total_unpinned.resize(num_devices);
  _to_free_tensors.resize(num_devices);
  _free_cache.resize(num_devices);
  _pinned_tensors.resize(num_devices);
  _total_to_free.resize(num_devices);
  _mem_pools.resize(num_devices);

  for (auto dev = 0; dev < num_devices; ++dev) {
    _total_free[dev] = mem_per_gpu;
    _total_unpinned[dev] = 0;
    _total_to_free[dev] = 0;
    _mem_pools[dev] = std::make_shared<gpu_memory_pool_t>(dev, mem_per_gpu);
  }
};

gpu_memory_t::~gpu_memory_t() {}

std::tuple<uint32_t, uint32_t, tid_t, std::shared_ptr<tensor_t>, bool>
gpu_memory_t::_try_to_find_gc(const kernel_prep_ptr_t &kp, 
                              size_t num_bytes,
                              int32_t dev) {  

  for (auto it = _unpinned_tensors[dev].begin(); it != _unpinned_tensors[dev].end(); ++it) {

    // if this is one of the input tensors skip
    if (std::find(kp->input.begin(), kp->input.end(), it->second) != kp->input.end()) {
      continue;
    }

    // did we find one
    auto &t = _tensors[it->second];
    if (t.num_bytes == num_bytes) {

      assert(t.data[dev] != nullptr);
      // of we found it erase it and return it
      std::tuple<uint32_t, uint32_t, tid_t, std::shared_ptr<tensor_t>, bool> out =
      {
        std::get<0>(it->first), 
        std::get<1>(it->first), 
        it->second,
        t.data[dev],
        t.num_copies <= 1 && !t.is_on_cpu && !t.is_evicting
      };

      t.unpinned_its[dev] = _unpinned_tensors[dev].end();
      _unpinned_tensors[dev].erase(it);
      return out;
    }
  }

  // we did not find anything return
  return {0, 0, -1, nullptr, false};
}

void gpu_memory_t::_mark_apply_for_use(const gpu_command_schedule_ptr_t &apply) {

  for(auto in_idx = 0; in_idx < apply->cmd->get_num_inputs(); in_idx++) {

    // get the tid
    auto tid = apply->cmd->get_inputs()[in_idx].tid;
    auto num_bytes = apply->input_sizes[in_idx];

    // get the tensors and update the use count
    bool created;
    auto &t = _init_tensor(tid, num_bytes, created);
    t.num_uses++;

    // update the unpinned tensor so that they are sorted right
    for (auto dev = 0; dev < _num_devices; ++dev) {

      // check if we actually have it
      if (t.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
        _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
        t.unpinned_its[dev] = _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, tid});
      }
    }
  }
};

void gpu_memory_t::_mark_reduce_for_use(const gpu_command_schedule_ptr_t &reduce) {

  for(auto in_idx = 0; in_idx < reduce->cmd->get_num_inputs(); in_idx++) {

    // get the tid
    auto tid = reduce->cmd->get_inputs()[in_idx].tid;
    auto num_bytes = reduce->input_sizes[in_idx];

    // get the tensors and update the use count
    bool created;
    auto t = _init_tensor(tid, num_bytes, created);
    t.num_uses++;

    // update the unpinned tensor so that they are sorted right
    for (auto dev = 0; dev < _num_devices; ++dev) {

      // check if we actually have it
      if (t.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
        _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
        t.unpinned_its[dev]  = _unpinned_tensors[dev].insert({{t.num_copies, 
                                                               t.num_uses}, tid});
      }
    }
  }
}

bool gpu_memory_t::mark_as_used(tid_t id) {

  // reduce the number of uses
  auto it = _tensors.find(id); assert(it != _tensors.end());
  it->second.num_uses--;

  // check if we have it if we do just remove it
  if(it->second.num_uses == 0 &&
     it->second.should_delete) {
    _remove(id, it->second.num_bytes);
    return true;
  }

  // update the unpinned tensor so that they are sorted right
  for (auto dev = 0; dev < _num_devices; ++dev) {

    // check if we actually have it
    if (it->second.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
      _unpinned_tensors[dev].erase(it->second.unpinned_its[dev]);
      it->second.unpinned_its[dev] = _unpinned_tensors[dev].insert({{it->second.num_copies, 
                                                                     it->second.num_uses}, id});
    }
  }
  return false;
};

void gpu_memory_t::mark_for_use(const gpu_command_schedule_ptr_t &cmd_sch) {
  assert(cmd_sch->cmd->is_apply() || cmd_sch->cmd->is_reduce());
  if(cmd_sch->cmd->is_apply()) {
    _mark_apply_for_use(cmd_sch);
  }
  else if(cmd_sch->cmd->is_reduce()){
    _mark_reduce_for_use(cmd_sch);
  }
}

bool gpu_memory_t::mark_for_deletion(tid_t id) {

  // find the tensor and make sure it was initialized previously
  auto it = _tensors.find(id);
  assert(it != _tensors.end() && it->second.num_bytes != 0);

  // if we are not using this tensor anymore we are free to delete it
  if(it->second.num_uses == 0) {
    _remove(id, it->second.num_bytes);
    return true;
  }

  // mark that we should delete this tensor as soon as can
  it->second.should_delete = true;
  return false;
};

void gpu_memory_t::_unpin_tensor(tid_t id, int dev, size_t num_bytes) {

  // grab it from the pinned tensors
  auto it = _pinned_tensors[dev].find(id);

  // decrement the count and check if it got unpinned
  if (--it->second == 0) {

    // remove it from the pinned tensors
    _pinned_tensors[dev].erase(it);

    // insert it into the unpinned tensors
    auto it = _tensors.find(id); assert(it != _tensors.end());
    it->second.unpinned_its[dev] =
        _unpinned_tensors[dev].insert({{it->second.num_copies, it->second.num_uses}, id});

    // increase the amount of unpinned memory
    _total_unpinned[dev] += num_bytes;
  }
};

void gpu_memory_t::pin_all(kernel_prep_ptr_t kp, int dev) {

  // this takes care of pinning all the input tensors
  for (auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
    _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
  }

  // this takes care of pinning all the output tensors
  for (auto out_idx = 0; out_idx < kp->input.size(); ++out_idx) {
    _pin_tensor(kp->output[out_idx], dev, kp->output_sizes[out_idx]);
  }
};

void gpu_memory_t::unpin_all(kernel_prep_ptr_t kp, int dev) {
  
  // this takes care of pinning all the input tensors
  for (auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
    _unpin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
  }

  // this takes care of pinning all the output tensors
  for (auto out_idx = 0; out_idx < kp->output.size(); ++out_idx) {
    _unpin_tensor(kp->output[out_idx], dev, kp->output_sizes[out_idx]);
  }
}

void gpu_memory_t::tensor_loaded_on_gpu(tid_t id, int dev, size_t num_bytes) {

  // init the tensor if necessary
  bool created;
  auto &t = _init_tensor(id, num_bytes, created);

  // mark that it is loaded
  t.is_loaded_on_gpu[dev] = true;
  assert(t.data[dev] != nullptr);
  t.num_copies++;

  // if the tensor was just created we mark it as unpinned
  if(created) {
    t.unpinned_its[dev] = _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, id});
  }
  // otherwise just update the stats if it is already unpinned
  else if(t.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
    _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
    t.unpinned_its[dev] = _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, id});
  }
};

void gpu_memory_t::tensor_loaded_on_cpu(tid_t id, size_t num_bytes) {

  // init the tensor if necessary
  bool created;
  _init_tensor(id, num_bytes, created);

  // mark that the tensor is now on the CPU
  auto it = _tensors.find(id); assert(it != _tensors.end());
  it->second.is_on_cpu = true;
}

int gpu_memory_t::can_preallocate(kernel_prep_ptr_t kp, 
                                  int32_t target_dev) {

  std::shared_ptr<tensor_t> tmp;
  std::vector<std::tuple<std::shared_ptr<tensor_t>, size_t>> tensors;                    
  for(auto dev = 0; dev < _num_devices; ++dev) {
    
    auto cur_dev = (dev + target_dev) % _num_devices;
    
    // reset the tensor ptrs
    kp->input_tensor_ptrs.clear();
    kp->input_tensor_ptrs.resize(kp->input.size());
    kp->output_tensor_ptrs.clear();
    kp->output_tensor_ptrs.resize(kp->output.size());

    bool success = true;
    for(auto out_idx = 0; out_idx < kp->output.size(); out_idx++) {
      auto out_size = kp->output_sizes[out_idx];
      tmp = _allocate_tensor(out_size, cur_dev);
      if(tmp == nullptr) {
          success = false;
          goto free_all;
      }
      kp->output_tensor_ptrs[out_idx] = tmp;
      tensors.push_back({std::move(tmp), out_size});
    }

    for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {

      auto in = kp->input[in_idx];
      auto in_size = kp->input_sizes[in_idx];

      if(!_is_on_device(in, cur_dev) && !_is_transfered_to_device(in, cur_dev)) {
        tmp = _allocate_tensor(in_size, cur_dev);
        if(tmp == nullptr) {
          success = false;
          goto free_all;
        }
        kp->input_tensor_ptrs[in_idx] = tmp;
        tensors.push_back({std::move(tmp), in_size});
      }
      else {
        kp->input_tensor_ptrs[in_idx] = nullptr;
      }
    }

    // if we have succeeded we are honkey-doray 
    if(success) {
      return cur_dev;
    }

free_all:

    // free everything so far
    for(auto &tmp : tensors) {
      auto &[t, num_bytes] = tmp;
      _mem_pools[cur_dev]->free(t->get_data_ptr<void*>(), num_bytes);
    }
    tensors.clear();
    continue;
  }

  return -1;
}

void gpu_memory_t::preallocate(kernel_prep_ptr_t kp, int32_t dev) {

  // set the device in the kernel prep
  cudaSetDevice(dev);
  kp->dev = dev;

  // sum all the output bytes
  size_t output_bytes_required = 0;
  for(auto out_idx = 0; out_idx < kp->output.size(); ++out_idx) {

    // init the tensor if it is not initialized
    bool created;
    auto tid = kp->output[out_idx];
    auto num_bytes = kp->output_sizes[out_idx];
    auto &t = _init_tensor(tid, num_bytes, created);

    // allocate it 
    t.data[dev] = kp->output_tensor_ptrs[out_idx];
    assert(t.data[dev] != nullptr);
    kp->run_me->outputs.set(out_idx, *t.data[dev]);

    // we just initialized and plan on filling it out during 
    // the kernel call therefore we need to pin the memory
    _total_free[dev] -= kp->output_sizes[out_idx];
    auto num_pinned = _pinned_tensors[dev][tid]++;
    assert(num_pinned == 0);
  }

  // go through each device and check if we can put it there
  for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {

    // check if it already present on the device
    auto src_dev = _is_on_any(kp->input[in_idx], dev);
    if(src_dev != dev) {
      
      // we also need to intiate a transfer from somewhere 
      // first check if there is an ongoing GPU to GPU transfer to this device
      bool created;
      auto &t = _init_tensor(kp->input[in_idx], kp->input_sizes[in_idx], created);

      if(t.gpu_transfers[dev] != nullptr) {

        // set the input from the transfer
        kp->run_me->inputs.set(in_idx, *t.gpu_transfers[dev]->dst);
        _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
        assert(t.gpu_transfers[dev]->dst != nullptr);

        // set the transfer to the kernel prep
        kp->gpu_transfers.push_back(t.gpu_transfers[dev]);
      }
      // next check if there is an ongoing CPU transfer to this device
      else if(t.cpu_transfer != nullptr && t.cpu_transfer->dst_dev == dev) {

        // set the input from the transfer
        kp->run_me->inputs.set(in_idx, *t.cpu_transfer->dst);
        _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
        assert(t.cpu_transfer->dst != nullptr);

        // set the transfer to the kernel prep
        assert(t.is_on_cpu || t.cpu_transfer->depends != nullptr);
        kp->cpu_transfers.push_back(t.cpu_transfer);
      }
      // next check if the tensor is on any GPU? so that we can fetch it from there
      else if (src_dev != -1) {

        // allocate the memory 
        t.data[dev] = kp->input_tensor_ptrs[in_idx];
        kp->run_me->inputs.set(in_idx, *t.data[dev]);
        assert(t.data[dev] != nullptr);

        // since we just created a tensor on this device
        // we need to mark the free memory as used 
        _total_free[dev] -= kp->input_sizes[in_idx];
        auto num_pinned = _pinned_tensors[dev][kp->input[in_idx]]++;
        assert(num_pinned == 0);

        // pin the source tensor
        _pin_tensor(kp->input[in_idx], src_dev, kp->input_sizes[in_idx]);

        // if it is on any other GPU make a GPU2GPU transfer
        auto transfer = std::make_shared<gpu_to_gpu_transfer_t>();

        // init the transfer
        transfer->id = _cur_gpu_tp_gpu_tranfer_id++;
        transfer->tid = kp->input[in_idx];
        transfer->src = _tensors[kp->input[in_idx]].data[src_dev];
        transfer->src_dev = src_dev;
        transfer->dst = t.data[dev];
        transfer->dst_dev = dev;
        transfer->num_bytes = kp->input_sizes[in_idx];
        transfer->depends = nullptr;

        // store the transfer
        _tensors[kp->input[in_idx]].gpu_transfers[dev] = transfer;
        _gpu_to_gpu_transfer[transfer->id] = transfer;

        // set the transfer
        t.gpu_transfers[dev] = transfer;
        kp->gpu_transfers.push_back(transfer);
      }
      // otherwise is the tensor currenly being transfered from CPU to any GPU?  
      else if (_tensors[kp->input[in_idx]].cpu_transfer != nullptr) {
        
        // we need to latch onto the CPU transfer
        src_dev = _tensors[kp->input[in_idx]].cpu_transfer->dst_dev;
        t.data[dev] = kp->input_tensor_ptrs[in_idx];
        kp->run_me->inputs.set(in_idx, *t.data[dev]);
        assert(t.data[dev] != nullptr);

        // since we just created a tensor on this device
        // we need to mark the free memory as used 
        _total_free[dev] -= kp->input_sizes[in_idx];
        auto num_pinned = _pinned_tensors[dev][kp->input[in_idx]]++;
        assert(num_pinned == 0);

        // pin an additional time the source tensor
        _pin_tensor(kp->input[in_idx], src_dev, kp->input_sizes[in_idx]);

        // if it is on any other GPU make a GPU2GPU transfer
        auto transfer = std::make_shared<gpu_to_gpu_transfer_t>();

        // init the transfer
        transfer->id = _cur_gpu_tp_gpu_tranfer_id++;
        transfer->tid = kp->input[in_idx];
        transfer->src = _tensors[kp->input[in_idx]].data[src_dev];
        transfer->src_dev = src_dev;
        transfer->dst = t.data[dev];
        transfer->dst_dev = dev;
        transfer->num_bytes = kp->input_sizes[in_idx];
        transfer->depends = _tensors[kp->input[in_idx]].cpu_transfer;

        // store the transfer
        _gpu_to_gpu_transfer[transfer->id] = transfer;
        t.gpu_transfers[dev] = transfer;
        kp->gpu_transfers.push_back(transfer);
      }
      // we don't have an ongoing CPU2GPU transfer we need to initiate one
      else {

        // allocate and pin the tensor
        t.data[dev] = kp->input_tensor_ptrs[in_idx];
        kp->run_me->inputs.set(in_idx, *t.data[dev]);
        assert(t.data[dev] != nullptr);

        // pin it since we are transfering this
        _total_free[dev] -= kp->input_sizes[in_idx];
        auto num_pinned = _pinned_tensors[dev][kp->input[in_idx]]++;
        assert(num_pinned == 0);

        // if it is on any other GPU make a CPU2GPU transfer
        auto transfer = std::make_shared<cpu_to_gpu_transfer_t>();

        // init the transfer
        transfer->id = _cur_cpu_to_gpu_transfer_id++;
        transfer->tid = kp->input[in_idx];
        transfer->dst = t.data[dev];
        transfer->dst_dev = dev;
        transfer->is_finished = false;
        transfer->num_bytes = kp->input_sizes[in_idx];
        transfer->depends = t.eviction_request;

        // store the transfer
        _cpu_to_gpu_transfer[transfer->id] = transfer;
        t.cpu_transfer = transfer;
        assert(t.is_on_cpu || transfer->depends != nullptr);
        kp->cpu_transfers.push_back(t.cpu_transfer);
      }
    }
    else {

      // set is in the krenel prep add another pin on this tensor
      kp->run_me->inputs.set(in_idx, *_tensors[kp->input[in_idx]].data[dev]);
      _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
      assert(_tensors[kp->input[in_idx]].data[dev] != nullptr);
    }
  }
};

void gpu_memory_t::finish_kernel_prep(kernel_prep_ptr_t kp, 
                                      std::vector<tid_t> &deleted_tensors) {

  // unpin input tensors
  assert(deleted_tensors.empty());
  for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
    _unpin_tensor(kp->input[in_idx], kp->dev, kp->input_sizes[in_idx]);
    if(mark_as_used(kp->input[in_idx])) {
      deleted_tensors.push_back(kp->input[in_idx]);
    }

    // check if this was an anonymous tensor
    if(kp->input[in_idx] < 0) {
      if(mark_for_deletion(kp->input[in_idx])) {
        deleted_tensors.push_back(kp->input[in_idx]);
      }
    }
  }

  // mark each output tensor as loaded and unpin it since we are not actively using it
  for(auto out_idx = 0; out_idx < kp->output.size(); ++out_idx) {
    tensor_loaded_on_gpu(kp->output[out_idx], kp->dev, kp->output_sizes[out_idx]);
    _unpin_tensor(kp->output[out_idx], kp->dev, kp->output_sizes[out_idx]);
  }
}


void gpu_memory_t::mark_transfer_done(cpu_to_gpu_transfer_ptr_t kp) {

  // if it is already retired we are good
  if(kp->is_retired) { return; }

  // make sure it is actually finished
  assert(kp->is_finished);

  // get the tensor
  auto &t = _tensors[kp->tid];

  // mark that it is finished and remove the transfer
  assert(t.cpu_transfer->id == kp->id);
  t.is_loaded_on_gpu[kp->dst_dev] = true;
  assert(t.data[kp->dst_dev] != nullptr);
  t.num_copies++;
  t.cpu_transfer = nullptr;
  _cpu_to_gpu_transfer.erase(kp->id);

  // mark as retired
  kp->is_retired = true;
}

void gpu_memory_t::mark_transfer_done(gpu_to_gpu_transfer_ptr_t kp) {

  // if it is already retired we are good
  if(kp->is_retired) { return; }

  // make sure it is actually finished
  assert(kp->is_finished);

  // unpin the source tensor
  _unpin_tensor(kp->tid, kp->src_dev, kp->num_bytes);

  // get the tensor
  auto &t = _tensors[kp->tid];

  // remove the transfer
  assert(t.gpu_transfers[kp->dst_dev]->id == kp->id);
  t.gpu_transfers[kp->dst_dev] = nullptr;
  t.is_loaded_on_gpu[kp->dst_dev] = true;

  // remove the gpu2gpu transfer
  _gpu_to_gpu_transfer.erase(kp->id);
  kp->is_retired = true;
}

gpu_memory_t::gc_approval_t gpu_memory_t::can_gc(kernel_prep_ptr_t kp, int32_t target_dev) { 

  std::shared_ptr<tensor_t> tmp;

  // we keep here a list of tensors we want to free if we fail to GC
  std::vector<std::tuple<std::shared_ptr<tensor_t>, size_t>> free_if_fail;  

  // all the tensors we pan to GC
  std::vector<std::tuple<uint32_t, uint32_t, tid_t, bool>> plan_to_gc;  

  // to unpin later
  std::vector<std::tuple<tid_t, size_t>> to_unpin;
  
  for(auto dev = 0; dev < _num_devices; ++dev) {

    // figure out the current device
    int32_t cur_dev = (dev + target_dev) % _num_devices;
    
    // free all tensors
    _free_all(cur_dev);

    // 1. allocate all the output tensors we can and if we can not try to find some to flush...
    kp->output_tensor_ptrs.clear();
    kp->output_tensor_ptrs.resize(kp->output.size());

    bool success = true;
    for(auto out_idx = 0; out_idx < kp->output.size(); out_idx++) {

      // try to allocate the tensor
      auto out_size = kp->output_sizes[out_idx];
      tmp = _allocate_tensor(out_size, cur_dev);

      // well we could not allocate try to find an unpinned tensor with the right size
      if(tmp == nullptr) {

        // try to GC this
        auto to_gc = _try_to_find_gc(kp, out_size, cur_dev);
        
        // if we failed we go and free all
        if(std::get<2>(to_gc) == -1) {
          goto free_all;
        }

        // try to GC this one
        tmp = std::get<3>(to_gc);
        plan_to_gc.push_back({std::get<0>(to_gc), 
                              std::get<1>(to_gc), 
                              std::get<2>(to_gc),
                              std::get<4>(to_gc)});
      }

      // set it
      kp->output_tensor_ptrs[out_idx] = tmp;
      free_if_fail.push_back({std::move(tmp), out_size});
    }

    // 2. allocate all the input tensors we can
    kp->input_tensor_ptrs.clear();
    kp->input_tensor_ptrs.resize(kp->input.size());
    for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {

      auto in = kp->input[in_idx];
      auto in_size = kp->input_sizes[in_idx];

      if(!_is_on_device(in, cur_dev) && !_is_transfered_to_device(in, cur_dev)) {

        tmp = _allocate_tensor(in_size, cur_dev);
        if(tmp == nullptr) {

          // try to GC this
          auto to_gc = _try_to_find_gc(kp, in_size, cur_dev);
          
          // if we failed we go and free all
          if(std::get<2>(to_gc) == -1) {
            goto free_all;
          }

          // try to GC this one
          tmp = std::get<3>(to_gc);
          plan_to_gc.push_back({std::get<0>(to_gc), 
                                std::get<1>(to_gc), 
                                std::get<2>(to_gc),
                                std::get<4>(to_gc)});
        }

        kp->input_tensor_ptrs[in_idx] = tmp;
        free_if_fail.push_back({std::move(tmp), in_size});
      }
      else {

        // we are going to later set this in prealloc
        kp->input_tensor_ptrs[in_idx] = nullptr;

        // pin the tensor and mark it later to unpin
        _pin_tensor(kp->input[in_idx], cur_dev, kp->input_sizes[in_idx]);
        to_unpin.push_back({kp->input[in_idx], kp->input_sizes[in_idx]});
      }
    }

    // we are good just return it
    return {.to_unpin = to_unpin,
            .free_if_fail = free_if_fail,
            .plan_to_gc = plan_to_gc, 
            .dev = cur_dev};

free_all:

    // unpin all the tensors that we pinned....
    for(auto u : to_unpin) {
      auto [tid, num_bytes] = u;
      _unpin_tensor(tid, cur_dev, num_bytes);
    }
    to_unpin.clear();

    // return all the unpinned tensors
    for(auto &gc : plan_to_gc) {
      _tensors[std::get<2>(gc)].unpinned_its[cur_dev] =
          _unpinned_tensors[cur_dev].insert(
              {{std::get<0>(gc), std::get<1>(gc)}, std::get<2>(gc)});
    }
    plan_to_gc.clear();

    // 3. if not free all the tensors and go to the next one
    for(auto &tmp : free_if_fail) {
      auto &[t, num_bytes] = tmp;
      _mem_pools[cur_dev]->free(t->get_data_ptr<void*>(), num_bytes);
    }
    free_if_fail.clear();
  }

  return gpu_memory_t::gc_approval_t{.free_if_fail = {}, .plan_to_gc = {}, .dev = -1};
};

gc_request_ptr_t gpu_memory_t::get_gc_request(kernel_prep_ptr_t kp, 
                                              gpu_memory_t::gc_approval_t &approval) {

  // make the request and set the device
  gc_request_ptr_t request = std::make_shared<gc_request_t>();
  request->dev = approval.dev;
  request->to_run = kp;
  request->to_unpin = std::move(approval.to_unpin);
  kp->dev = approval.dev;

  // setup all the tensors we want to evict
  for(auto to_evict : approval.plan_to_gc) {
    
    // ge the tensor
    auto tid = std::get<2>(to_evict);
    auto need_evict = std::get<3>(to_evict);
    auto &t = _tensors[tid];

    // setup the request
    if(need_evict) {

      // create an eviction request
      assert(!_tensors[tid].is_on_cpu);
      request->to_evict.push_back(std::make_shared<gc_request_evict_t>());
      request->to_evict.back()->evicted = false;
      request->to_evict.back()->num_bytes = t.num_bytes;
      assert(t.data[kp->dev] != nullptr);
      request->to_evict.back()->tensor = t.data[kp->dev];
      request->to_evict.back()->tid = tid;
      t.eviction_request = request->to_evict.back();
      t.is_evicting = true;
    }

    // remove the tensor we just evicted and set the eviction request
    t.data[kp->dev] = nullptr;
    t.num_copies -= 1;
    t.is_loaded_on_gpu[kp->dev] = false;
    t.unpinned_its[kp->dev] = _unpinned_tensors[kp->dev].end();

    for(auto fc : _free_cache[kp->dev]) {
      assert(fc.second != tid);
    }

    // we used the unpinned memory
    _total_unpinned[kp->dev] -= t.num_bytes;
  }

  // make sure it is 
  return std::move(request);
};

void gpu_memory_t::finish_gc_request(const gc_request_ptr_t &req) {

  // null out the eviction requests
  for(auto to_evict : req->to_evict) {
    _tensors[to_evict->tid].is_on_cpu = true;
    _tensors[to_evict->tid].is_evicting = false;
    _tensors[to_evict->tid].eviction_request = nullptr;
  }

  // unpin all the tensors that we pinned....
  for(auto u : req->to_unpin) {
    auto [tid, num_bytes] = u;
    _unpin_tensor(tid, req->dev, num_bytes);
  }
}

bool gpu_memory_t::get_tensors_to_flush(std::vector<std::tuple<tensor_t*, tid_t, size_t>> &to_flush) {

  bool can_flush_all = true;
  for(auto &t : _tensors) {

    // this is here to make sure all the tensors are available
    can_flush_all = can_flush_all && (t.second.num_copies > 0 || t.second.is_on_cpu); 

    // if this one is just on the GPU but not on the CPU we need to flush it
    for(auto dev = 0; dev < _num_devices; dev++) {
      if(!t.second.is_on_cpu && t.second.is_loaded_on_gpu[dev])  {
        to_flush.push_back({t.second.data[dev].get(), t.first, t.second.num_bytes});
      }
    }
  }
  return can_flush_all;
}

void gpu_memory_t::mark_as_flushed(const std::vector<std::tuple<tensor_t*, tid_t, size_t>> &to_flush) {

  // go and mark each tensor...
  for(auto &t : to_flush) {
    auto tid = std::get<1>(t);
    auto it = _tensors.find(tid); assert(it != _tensors.end());
    it->second.is_on_cpu = true;
  }
}

void gpu_memory_t::_remove(tid_t id, size_t num_bytes) {

  // free this tensor now as it is not needed anymore
  for(int32_t dev = 0; dev < _num_devices; ++dev) {
    auto it = _tensors.find(id); assert(it != _tensors.end());
    if(it->second.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
      
      // remove it from the unpinned tensors
      _unpinned_tensors[dev].erase(it->second.unpinned_its[dev]);
      it->second.unpinned_its[dev] = _unpinned_tensors[dev].end();

      // make sure we don't have a ongoing transfer
      assert(it->second.cpu_transfer == nullptr);
      for(auto &t : it->second.gpu_transfers) { assert(t == nullptr); }

      // add it to the free
      _to_free_tensors[dev].push_back(id);

      // store it in the cache
      assert(it->second.data[dev] != nullptr);
      it->second.free_its[dev] = _free_cache[dev].insert({num_bytes, id});

      // update the numbers
      _total_unpinned[dev] -= num_bytes;
      _total_to_free[dev] += num_bytes;

      // mark sure it is marked as not loaded
      it->second.is_loaded_on_gpu[dev] = false;

      // increment the number of tensors left to remove
      it->second.num_left_to_remove++;
    }

    // make sure it is not pinned anywhere
    assert(_pinned_tensors[dev].find(id) == _pinned_tensors[dev].end()); 
  }
}

void gpu_memory_t::_free_all(int32_t dev) {

  // free all the tensors we can free
  for (auto it = _free_cache[dev].begin(); it != _free_cache[dev].end(); ++it) {

    // we got a chuck from the free cache we should be able to use it...
    auto t = _tensors.find(it->second);

    // claim this memory
    _total_to_free[dev] -= t->second.num_bytes;
    _total_free[dev] += t->second.num_bytes;

    // null the data just in case
    assert(t->second.data[dev] != nullptr);
    auto kmp = t->second.data[dev]->get_data_ptr<void*>();
    t->second.data[dev] = nullptr;

    // we are killing a copy of this
    if(--t->second.num_left_to_remove == 0) {
      _tensors.erase(t);
    }

    // remove it from the free tensors
    auto jt = std::find(_to_free_tensors[dev].begin(), 
                        _to_free_tensors[dev].end(), 
                        it->second);
    std::iter_swap(jt, (_to_free_tensors[dev].end() - 1));
    _to_free_tensors[dev].pop_back();

    // free it from the memory pool
    _mem_pools[dev]->free(kmp, t->second.num_bytes);
  }

  // clear all the free on this device
  _free_cache[dev].clear();
}

gpu_memory_t::gpu_mem_tensor_t &gpu_memory_t::_init_tensor(tid_t id, 
                                                           size_t num_bytes, 
                                                           bool &created) {

  // check if we already have it
  auto it = _tensors.find(id);
  if(it != _tensors.end()) {
    assert(it->second.num_bytes == num_bytes);
    created = false;
    return it->second;
  }

  // init the rest and return
  auto &t = _tensors[id];
  t.should_delete = false;
  t.num_bytes = num_bytes;
  t.num_copies = 0;
  t.num_left_to_remove = 0;

  // anonymous tensors will always used once
  t.num_uses = id < 0 ? 1 : 0;
  t.cpu_transfer = nullptr;
  for(auto dev = 0; dev < _num_devices; ++dev) {
    t.unpinned_its[dev] = _unpinned_tensors[dev].end();
    t.free_its[dev] = _free_cache[dev].end();
    t.gpu_transfers[dev] = nullptr;
    t.data[dev] = nullptr;
  }
  created = true;
  return t;
}

std::shared_ptr<tensor_t> gpu_memory_t::_allocate_tensor(size_t num_bytes, int32_t dev) {

  // the allocated memory on the GPU
  void *tmp;

  // try to find it in the free cache (fast path)
  auto it = _free_cache[dev].find(num_bytes);
  if(it != _free_cache[dev].end()) {

    // we got a chuck from the free cache we should be able to use it...
    auto t = _tensors.find(it->second);

    // claim this memory
    _total_to_free[dev] -= t->second.num_bytes;
    _total_free[dev] += t->second.num_bytes;

    // null the data just in case
    assert(t->second.data[dev] != nullptr);
    tmp = t->second.data[dev]->get_data_ptr<void*>();
    assert(tmp != nullptr);
    t->second.data[dev] = nullptr;

    // we are killing a copy of this
    if(--t->second.num_left_to_remove == 0) {
      _tensors.erase(t);
    }

    // remove it from the free tensors
    auto jt = std::find(_to_free_tensors[dev].begin(), 
                        _to_free_tensors[dev].end(), 
                        it->second);
    std::iter_swap(jt, (_to_free_tensors[dev].end() - 1));
    _to_free_tensors[dev].pop_back();

    // remove it form the cache
    _free_cache[dev].erase(it);
  }
  else {
    
    // slow path
    tmp = _mem_pools[dev]->allocate(num_bytes);
  }

  // ok we can allocate space in any way lets 
  if(tmp == nullptr) {

    // free all tensors
    _free_all(dev);

    // try to allocate again
    tmp = _mem_pools[dev]->allocate(num_bytes);
  }

  return tmp == nullptr ? nullptr : std::move(std::make_shared<tensor_t>(tmp));
}

bool gpu_memory_t::_is_pinned(tid_t id, int32_t dev) {
  return _pinned_tensors[dev].find(id) != _pinned_tensors[dev].end();
}

bool gpu_memory_t::_is_unpinned(tid_t id, int32_t dev) {
  auto it = _tensors.find(id); assert(it != _tensors.end());
  return it->second.unpinned_its[dev] != _unpinned_tensors[dev].end();
}

bool gpu_memory_t::_is_on_device(tid_t id, int32_t dev) {
  auto it = _tensors.find(id); assert(it != _tensors.end());
  return _tensors[id].is_loaded_on_gpu[dev];
}

bool gpu_memory_t::_is_transfered_to_device(tid_t id, int32_t dev) {
  auto it = _tensors.find(id); assert(it != _tensors.end());
  return it->second.gpu_transfers[dev] != nullptr ||
        (it->second.cpu_transfer != nullptr && it->second.cpu_transfer->dst_dev == dev);
}

int32_t gpu_memory_t::_is_on_any(tid_t id, int32_t target_dev) {
  auto it = _tensors.find(id); assert(it != _tensors.end());
  for(int32_t dev = 0; dev < _num_devices; dev++) {
    auto cur_dev = (dev + target_dev) % _num_devices;
    if(it->second.is_loaded_on_gpu[cur_dev]) {
      return cur_dev;
    }
  }
  return -1;
}

void gpu_memory_t::_pin_tensor(tid_t id, int32_t dev, size_t num_bytes) {

  // check if this is unpinned? if it is we need to remove it from there
  auto it = _tensors.find(id); assert(it != _tensors.end());
  if (it->second.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
    _unpinned_tensors[dev].erase(it->second.unpinned_its[dev]);
    it->second.unpinned_its[dev] = _unpinned_tensors[dev].end();
  }

  // pin it (increment the count and in the case if does not exist it will create it)
  auto &num_pinned = _pinned_tensors[dev][id];

  // if this is just getting pinned
  if(num_pinned == 0) {
    _total_unpinned[dev] -= num_bytes;
  }

  // increment the pin count
  num_pinned++;
}

} // namespace bbts