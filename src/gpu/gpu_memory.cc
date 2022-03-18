#include "gpu_memory.h"
#include <cstddef>
#include <cstdint>

namespace bbts {

gpu_memory_t::gpu_memory_t(size_t num_devices, size_t mem_per_gpu)
    : _num_devices(num_devices) {

  _unpinned_tensors.resize(num_devices);
  _total_unpinned.resize(num_devices);
  _to_free_tensors.resize(num_devices);

  for (auto dev = 0; dev < num_devices; ++dev) {
    _total_free[dev] = mem_per_gpu;
    _total_unpinned[dev] = mem_per_gpu;
  }
};

void gpu_memory_t::mark_for_use(tid_t id) {

  // get the tensors and update the use count
  auto &t = _tensors[id];
  t.num_uses++;

  // update the unpinned tensor so that they are sorted right
  for (auto dev = 0; dev < _num_devices; ++dev) {

    // check if we actually have it
    if (t.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
      _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
      _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, id});
    }
  }
};

void gpu_memory_t::mark_as_used(tid_t id) {

  // reduce the number of uses
  auto &t = _tensors[id];
  t.num_uses--;

  // update the unpinned tensor so that they are sorted right
  for (auto dev = 0; dev < _num_devices; ++dev) {

    // check if we actually have it
    if (t.unpinned_its[dev] != _unpinned_tensors[dev].end()) {
      _unpinned_tensors[dev].erase(t.unpinned_its[dev]);
      _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, id});
    }
  }
};

void gpu_memory_t::mark_for_deletion(tid_t id){

};

void gpu_memory_t::_unpin_tensor(tid_t id, int dev, size_t num_bytes) {

  // grab it from the pinned tensors
  auto it = _pinned_tensors[dev].find(id);

  // decrement the count and check if it got unpinned
  if (--it->second == 0) {

    // remove it from the pinned tensors
    _pinned_tensors[dev].erase(it);

    // insert it into the unpinned tensors
    auto &t = _tensors[id];
    t.unpinned_its[dev] =
        _unpinned_tensors[dev].insert({{t.num_copies, t.num_uses}, id});

    // 
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
  for (auto out_idx = 0; out_idx < kp->input.size(); ++out_idx) {
    _unpin_tensor(kp->output[out_idx], dev, kp->output_sizes[out_idx]);
  }
}

void gpu_memory_t::tensor_created(tid_t id, int dev){

};

int gpu_memory_t::can_preallocate(kernel_prep_ptr_t kp) { 

  // sum all the output bytes
  size_t output_bytes_required = 0;
  for(auto out_size : kp->output_sizes) {
    output_bytes_required += out_size;
  }

  // go through each device and check if we can put it there
  for(auto dev = 0; dev < _num_devices; ++dev) {
    size_t required = output_bytes_required; 
    for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
      if(!_is_on_device(kp->input[in_idx], dev)) {
        required += kp->input_sizes[in_idx];
      }
    }
    
    // do we have enough memory
    if(_total_free[dev] >= required) {
      return dev;
    }
  }

  // we can not preallocate
  return -1; 
};

void gpu_memory_t::mark_transfer_done(cpu_to_gpu_transfer_ptr_t kp) {

}

void gpu_memory_t::mark_transfer_done(gpu_to_gpu_transfer_ptr_t kp) {

}

void gpu_memory_t::preallocate(kernel_prep_ptr_t kp, int dev){};

int gpu_memory_t::can_gc(kernel_prep_ptr_t kp) { return -1; };

gc_request_ptr_t gpu_memory_t::get_gc_request(kernel_prep_ptr_t kp, int dev) {
  return nullptr;
};

bool gpu_memory_t::_is_on_device(tid_t id, int32_t dev) {
  return _tensors[id].unpinned_its[dev] != _unpinned_tensors[dev].end() || 
         _pinned_tensors[dev].find(id) != _pinned_tensors[dev].end();
}

void gpu_memory_t::_pin_tensor(tid_t id, int32_t dev, size_t num_bytes) {

  // check if this is unpinned? if it is we need to remove it from there
  if (_tensors[dev].unpinned_its[dev] != _unpinned_tensors[dev].end()) {
    _unpinned_tensors[dev].erase(_tensors[dev].unpinned_its[dev]);
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