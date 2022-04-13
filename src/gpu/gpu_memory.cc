#include "gpu_memory.h"
#include "types.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <memory>
#include <vector>

namespace bbts {

gpu_memory_t::gpu_memory_t(size_t num_devices, size_t mem_per_gpu)
    : _num_devices(num_devices) {

  _unpinned_tensors.resize(num_devices);
  _total_free.resize(num_devices);
  _total_unpinned.resize(num_devices);
  _to_free_tensors.resize(num_devices);
  _pinned_tensors.resize(num_devices);

  for (auto dev = 0; dev < num_devices; ++dev) {
    _total_free[dev] = mem_per_gpu;
    _total_unpinned[dev] = 0;
  }
};

gpu_memory_t::~gpu_memory_t() {

  // free all the tensors
  for(auto &t : _tensors) {
    for(auto dev = 0; dev < _num_devices; ++dev) {
      if(t.second.data[dev] != nullptr) {
        cudaFree(t.second.data[dev]->get_data_ptr<void>());
      }
    }
  }
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

void gpu_memory_t::mark_as_used(tid_t id) {

  // reduce the number of uses
  auto it = _tensors.find(id); assert(it != _tensors.end());
  it->second.num_uses--;

  // check if we have it if we do just remove it
  if(it->second.num_uses == 0 &&
     it->second.should_delete) {
    _remove(id, it->second.num_bytes);
    return;
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

void gpu_memory_t::mark_for_deletion(tid_t id) {

  // find the tensor and make sure it was initialized previously
  auto it = _tensors.find(id);
  assert(it != _tensors.end() && it->second.num_bytes != 0);

  // if we are not using this tensor anymore we are free to delete it
  if(it->second.num_uses == 0) {
    _remove(id, it->second.num_bytes);
  }
  else {
    // mark that we should delete this tensor as soon as can
    it->second.should_delete = true;
  }
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
  return _fits_memory(kp, target_dev, [&](int32_t dev){
    return _total_free[dev];
  });
};

void gpu_memory_t::preallocate(kernel_prep_ptr_t kp, int32_t dev) {

  // set the device in the kernel prep
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
    t.data[dev] = _allocate_tensor(num_bytes);
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

        // set the transfer to the kernel prep
        kp->gpu_transfers.push_back(t.gpu_transfers[dev]);
      }
      // next check if there is an ongoing CPU transfer to this device
      else if(t.cpu_transfer != nullptr && t.cpu_transfer->dst_dev == dev) {

        // set the input from the transfer
        kp->run_me->inputs.set(in_idx, *t.cpu_transfer->dst);
        _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);

        // set the transfer to the kernel prep
        kp->cpu_transfers.push_back(t.cpu_transfer);
      }
      // next check if the tensor is on any GPU? so that we can fetch it from there
      else if (src_dev != -1) {

        // allocate the memory 
        t.data[dev] = _allocate_tensor(kp->input_sizes[in_idx]);
        kp->run_me->inputs.set(in_idx, *t.data[dev]);

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
        transfer->src = _tensors[kp->input[in_idx]].data[src_dev].get();
        transfer->src_dev = src_dev;
        transfer->dst = t.data[dev].get();
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
        t.data[dev] = _allocate_tensor(kp->input_sizes[in_idx]);
        kp->run_me->inputs.set(in_idx, *t.data[dev]);

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
        transfer->src = _tensors[kp->input[in_idx]].data[src_dev].get();
        transfer->src_dev = src_dev;
        transfer->dst = t.data[dev].get();
        transfer->num_bytes = kp->input_sizes[in_idx];
        transfer->depends = _tensors[kp->input[in_idx]].cpu_transfer;

        // store the transfer
        _tensors[kp->input[in_idx]].gpu_transfers[dev] = transfer;
        _gpu_to_gpu_transfer[transfer->id] = transfer;
        t.gpu_transfers[dev] = transfer;
        kp->gpu_transfers.push_back(transfer);
      }
      // we don't have an ongoing CPU2GPU transfer we need to initiate one
      else {

        // allocate and pin the tensor
        t.data[dev] = _allocate_tensor(kp->input_sizes[in_idx]);
        kp->run_me->inputs.set(in_idx, *t.data[dev]);

        // pin it since we are transfering this
        _total_free[dev] -= kp->input_sizes[in_idx];
        auto num_pinned = _pinned_tensors[dev][kp->input[in_idx]]++;
        assert(num_pinned == 0);

        // if it is on any other GPU make a GPU2GPU transfer
        auto transfer = std::make_shared<cpu_to_gpu_transfer_t>();

        // init the transfer
        transfer->id = _cur_cpu_to_gpu_transfer_id++;
        transfer->tid = kp->input[in_idx];
        transfer->dst = t.data[dev].get();
        transfer->dst_dev = dev;
        transfer->is_finished = false;
        transfer->num_bytes = kp->input_sizes[in_idx];

        // store the transfer
        _tensors[kp->input[in_idx]].cpu_transfer = transfer;
        _cpu_to_gpu_transfer[transfer->id] = transfer;
        t.cpu_transfer = transfer;
        kp->cpu_transfers.push_back(t.cpu_transfer);
      }
    }
    else {

      // set is in the krenel prep add another pin on this tensor
      kp->run_me->inputs.set(in_idx, *_tensors[kp->input[in_idx]].data[dev]);
      _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
    }
  }
};

void gpu_memory_t::finish_kernel_prep(kernel_prep_ptr_t kp) {

  // unpin input tensors
  for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
    _unpin_tensor(kp->input[in_idx], kp->dev, kp->input_sizes[in_idx]);
    mark_as_used(kp->input[in_idx]);

    // check if this was an anonymous tensor
    if(kp->input[in_idx] < 0) {
      mark_for_deletion(kp->input[in_idx]);
    }
  }

  // mark each output tensor as loaded and unpin it since we are not actively using it
  for(auto out_idx = 0; out_idx < kp->output.size(); ++out_idx) {
    tensor_loaded_on_gpu(kp->output[out_idx], kp->dev, kp->output_sizes[out_idx]);
    _unpin_tensor(kp->output[out_idx], kp->dev, kp->output_sizes[out_idx]);
  }
}


void gpu_memory_t::mark_transfer_done(cpu_to_gpu_transfer_ptr_t kp) {

  // make sure it is actually finished
  assert(kp->is_finished);

  // get the tensor
  auto &t = _tensors[kp->tid];

  // mark that it is finished and remove the transfer
  assert(t.cpu_transfer->id == kp->id);
  t.is_loaded_on_gpu[kp->dst_dev] = true;
  t.num_copies++;
  t.cpu_transfer = nullptr;
  _cpu_to_gpu_transfer.erase(kp->id);
}

void gpu_memory_t::mark_transfer_done(gpu_to_gpu_transfer_ptr_t kp) {

  // make sure it is actually finished
  assert(kp->is_finished);

  // unpin the destination tensor
  _unpin_tensor(kp->tid, kp->dst_dev, kp->num_bytes);

  // get the tensor
  auto &t = _tensors[kp->tid];

  // remove the transfer
  assert(t.gpu_transfers[kp->dst_dev]->id == kp->id);
  t.gpu_transfers[kp->dst_dev] = nullptr;

  // remove the gpu2gpu transfer
  _gpu_to_gpu_transfer.erase(kp->id);
}

int gpu_memory_t::can_gc(kernel_prep_ptr_t kp, int32_t target_dev) { 
  return _fits_memory(kp, target_dev, [&](int32_t dev){
    return _total_unpinned[dev] + _total_free[dev];
  });
};

gc_request_ptr_t gpu_memory_t::get_gc_request(kernel_prep_ptr_t kp, int dev) {

  // sum all the output bytes
  gc_request_ptr_t request = std::make_shared<gc_request_t>();
  size_t output_bytes_required = 0;
  for(auto out_size : kp->output_sizes) {
    output_bytes_required += out_size;
  }

  // go through each device and check if we can put it there
  int64_t required = output_bytes_required; 
  for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
    if(!_is_on_device(kp->input[in_idx], dev) &&
       !_is_transfered_to_device(kp->input[in_idx], dev)) {
      required += kp->input_sizes[in_idx];
    }
    else {
      _pin_tensor(kp->input[in_idx], dev, kp->input_sizes[in_idx]);
      request->to_unpin.push_back({kp->input[in_idx], kp->input_sizes[in_idx]});
    }
  }

  // make sure we actually have space
  assert((_total_unpinned[dev] + _total_free[dev]) >= required);

  // go through all the tensors we need to free
  for(int64_t idx = _to_free_tensors[dev].size() - 1; idx >= 0; --idx) {
    
    // are we done?
    if(required <= 0) {
      break;
    }

    // we are freeing this one
    auto &free_me = _to_free_tensors[dev][idx]; 
    
    // decrement the memory
    auto it = _tensors.find(free_me);
    auto &t = it->second;
    required -= t.num_bytes;

    // remove it from the structure
    _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
    t.unpinned_its[dev] = _unpinned_tensors[dev].end();

    // claim this memory
    _total_unpinned[dev] -= t.num_bytes;
    request->to_free.push_back({t.data[dev].get(), free_me, t.num_bytes});
    t.data[dev] = nullptr;

    // we are killing a copy of this
    t.is_loaded_on_gpu[dev] = false;
    if(--t.num_copies == 0) {
      _tensors.erase(it);
    }

    // remove the last one
    _to_free_tensors[dev].pop_back();
  }

  // go through all the tensors we need to evict
  while (true) {

    // are we done?
    if(required <= 0) {
      break;
    }
    
    // this is never supposed to happen because we called @see can_gc.
    assert(!_unpinned_tensors[dev].empty());

    // decrement the required memory
    auto evict = _unpinned_tensors[dev].begin();
    auto &t = _tensors[evict->second];
    required -= t.num_bytes;

    // do we have to evict it or can we kill it
    if(!t.is_on_cpu && t.num_copies == 1) {
      request->to_evict.push_back({t.data[dev].get(), evict->second, t.num_bytes});
    }
    else {
      request->to_free.push_back({t.data[dev].get(), evict->second, t.num_bytes});
    }

    // claim this memory
    _total_unpinned[dev] -= t.num_bytes;
    t.data[dev] = nullptr;

    // we just killed a copy
    t.num_copies--;
    t.is_loaded_on_gpu[dev] = false;

    // remove it from the unpinned list
    _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
    t.unpinned_its[dev] = _unpinned_tensors[dev].end();
  }

  // the kernel to run
  request->to_run = kp;
  request->dev = dev;

  // make sure it is 
  assert(required <= 0);
  return std::move(request);
};

void gpu_memory_t::finish_gc_request(const gc_request_ptr_t &req) {

  // increase the free memeory for each tensor we have freed
  for(auto r : req->to_free) {
    auto [_, tid, num_bytes] = r;
    _total_free[req->dev] += num_bytes;
  }

  // increase the free memory for each tensor we have evicted
  for(auto r : req->to_evict) {
    auto [_, tid, num_bytes] = r;
    _total_free[req->dev] += num_bytes;
  }

  // unpin all the tensors
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

      // make sure we don't have a ongoing transfer
      assert(it->second.cpu_transfer == nullptr);
      for(auto &t : it->second.gpu_transfers) { assert(t == nullptr); }

      // add it to the free
      _to_free_tensors[dev].push_back(id);

      // update the numbers
      _total_free[dev] += num_bytes;
      _total_unpinned[dev] -= num_bytes;
    }

    // make sure it is not pinned anywhere
    assert(_pinned_tensors[dev].find(id) == _pinned_tensors[dev].end()); 
  }
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

  // anonymous tensors will always used once
  t.num_uses = id < 0 ? 1 : 0;
  t.cpu_transfer = nullptr;
  for(auto dev = 0; dev < _num_devices; ++dev) {
    t.unpinned_its[dev] = _unpinned_tensors[dev].end();
    t.gpu_transfers[dev] = nullptr;
    t.data[dev] = nullptr;
  }
  created = true;
  return t;
}

std::shared_ptr<tensor_t> gpu_memory_t::_allocate_tensor(size_t num_bytes) {
  void *tmp;
  cudaMalloc(&tmp, num_bytes - sizeof(tensor_t));
  return std::move(std::make_shared<tensor_t>(tmp));
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
      return target_dev;
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