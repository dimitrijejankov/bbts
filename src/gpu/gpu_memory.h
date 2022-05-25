#include "types.h"
#include "gpu_memory_pool.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bbts {

// the gpu memory take care of the following three things:
// 1. it does preallocation of the GPU memory
// 2. it issues garbage collection requests in the case that we need to free some memory
// 3. it issues transfers to the kernel preps
class gpu_memory_t {
public:

  gpu_memory_t(size_t num_devices, size_t mem_per_gpu);

  ~gpu_memory_t();

  // mark that a tensor has already been used, returns tro if it should be removed 
  [[nodiscard]] bool mark_as_used(tid_t id);

  // mark the gpu command for use
  void mark_for_use(const gpu_command_schedule_ptr_t &cmd_sch);

  // mark this tensor for deletion returns true if the tensors is removed
  [[nodiscard]] bool mark_for_deletion(tid_t id);

  // pins all the tensors in the kernel prep
  void pin_all(kernel_prep_ptr_t kp, int dev);

  // unpins all the tensors in the kernel prep
  void unpin_all(kernel_prep_ptr_t kp, int dev);
  
  // mark that a tensor is not loaded and on a particular device
  // must be called when  
  void tensor_loaded_on_gpu(tid_t id, int dev, size_t num_bytes);

  // mark that the tensor is loaded on the CPU
  void tensor_loaded_on_cpu(tid_t id, size_t num_bytes);

  // can we preallocate all the tensors that we need to run the kernel
  int can_preallocate(kernel_prep_ptr_t kp, int32_t target_dev);

  // mark that a cpu transfer is done
  void mark_transfer_done(cpu_to_gpu_transfer_ptr_t kp);

  // mark that a gpu transfer is done
  void mark_transfer_done(gpu_to_gpu_transfer_ptr_t kp);

  // go through all the GPUs
  //   can we find enough memory on the GPU? (we find the memory in the
  //   following order)
  //   - can we delete some tensors that are scheduled for deletion
  //   - can we remove some redundant tensors that are already on the GPU.
  //   - is there any unpinned memory that we can
  // if there are multiple GPUs with resources we pick the one that needs to
  // transfer the least, if they need to transfer the same we pick the most
  // free memory (or based on the workload not sure) 
  void preallocate(kernel_prep_ptr_t kp, int dev);

  // finish the kernel prep
  void finish_kernel_prep(kernel_prep_ptr_t kp, 
                          std::vector<tid_t> &deleted_tensors);

  // can an we run garbage collection
  // if we can it returns the device othewise it returns -1
  int can_gc(kernel_prep_ptr_t kp, int32_t target_dev);

  // get the garbage collection request
  gc_request_ptr_t get_gc_request(kernel_prep_ptr_t kp, int dev);

  // finish the garbage collection request
  void finish_gc_request(const gc_request_ptr_t &req);

  // get all the tensors we can flush, returns true if all tensors can be flushed
  bool get_tensors_to_flush(std::vector<std::tuple<tensor_t*, tid_t, size_t>> &to_flush);

  // mark that all of these tensors were just flushed
  void mark_as_flushed(const std::vector<std::tuple<tensor_t*, tid_t, size_t>> &to_flush);

  // returns all the tensors that were deleted in the mean time
  std::vector<tid_t> get_deleted_tensors();

private:

  // mark that we are going to use all the inputs from the apply
  void _mark_apply_for_use(const gpu_command_schedule_ptr_t &apply);

  // mark that we are going to use all the inputs from the reduce
  void _mark_reduce_for_use(const gpu_command_schedule_ptr_t &reduce);

  // performs the actual remove
  void _remove(tid_t id, size_t num_bytes);
  
  // sorted by num_copies (first), num_uses (second)
  using unpinned_t = std::multimap<std::tuple<uint32_t, uint32_t>, tid_t>;

  struct gpu_mem_tensor_t {

    // is tensor on the CPU
    bool is_on_cpu = false;

    // make sure this is set when it is marked as deleted
    bool should_delete = true;
    
    // the number of uses of this tensor
    uint32_t num_uses = 0;

    // how many copies of this tensor are there in memory
    uint32_t num_copies = 0;

    // the size of the tensor
    size_t num_bytes = 0;

    // the number of tensors left to remove
    uint32_t num_left_to_remove = 0;

    // a pointer to the tensor
    std::array<std::shared_ptr<tensor_t>, BBTS_MAX_GPU_DEVICES> data;

    // is the tensor loaded on a particular device (a tensor is loaded once the transfer is finished)
    std::array<bool, BBTS_MAX_GPU_DEVICES> is_loaded_on_gpu;

    // the eviction requests that are currently going on
    gc_request_evict_ptr_t eviction_request;

    // these are iterators so that we can quickly update the _unpinned_tensors
    std::array<unpinned_t::iterator, BBTS_MAX_GPU_DEVICES> unpinned_its;

    // iterators to the free cache
    std::array<std::unordered_multimap<size_t, tid_t>::iterator, BBTS_MAX_GPU_DEVICES> free_its;

    // the ongoing transfers from GPU2GPU
    std::array<gpu_to_gpu_transfer_ptr_t, BBTS_MAX_GPU_DEVICES> gpu_transfers;
    
    // the transfers from CPU to GPU
    cpu_to_gpu_transfer_ptr_t cpu_transfer;
  };

  // initialize the tensor
  gpu_mem_tensor_t &_init_tensor(tid_t id, size_t num_bytes, bool &created);

  // allocate the tensor
  std::shared_ptr<tensor_t> _allocate_tensor(size_t num_bytes, int32_t dev);

  // is the tensor pinned
  bool _is_pinned(tid_t id, int32_t dev);

  // is the tensor unpinned
  bool _is_unpinned(tid_t id, int32_t dev);

  // check whether the tensor is present on the device (must have finished the transfer)
  bool _is_on_device(tid_t id, int32_t dev);

  // is it currently transfered 
  bool _is_transfered_to_device(tid_t id, int32_t dev);

  // check whether the tensor is present on any device 
  // the fuction will return the target_dev if it is on it otherwise it 
  // will return the current 
  int32_t _is_on_any(tid_t id, int32_t target_dev);

  // pin the tensor
  void _pin_tensor(tid_t id, int32_t dev, size_t num_bytes);

  // mark that a tensor is not unpinned
  void _unpin_tensor(tid_t id, int dev, size_t num_bytes);

  // used to implement both @see can_gc and @see can_preallocate
  template<class mem_pool_fun_t, class criteria_fun_t>
  int32_t _fits_memory(kernel_prep_ptr_t kp, 
                       int32_t target_dev, 
                       mem_pool_fun_t mem_pool_fun, 
                       criteria_fun_t criteria_fun) {
  
    // sum all the output bytes
    size_t output_bytes_required = 0;
    for(auto out_size : kp->output_sizes) {
      output_bytes_required += out_size;
    }

    // go through each device and check if we can put it there
    for(auto dev = 0; dev < _num_devices; ++dev) {
      auto cur_dev = (dev + target_dev) % _num_devices;
      size_t required = output_bytes_required; 
      for(auto in_idx = 0; in_idx < kp->input.size(); ++in_idx) {
        if(criteria_fun(kp->input[in_idx], cur_dev)) {
          required += kp->input_sizes[in_idx];
        }
      }
      
      // do we have enough memory
      auto mem_available = mem_pool_fun(cur_dev);
      if(mem_available >= required) {
        return cur_dev;
      }
    }

    // we can not preallocate
    return -1; 
  }

  // how many times will this tensor be used
  std::unordered_map<tid_t, gpu_mem_tensor_t> _tensors;

  // these are the pinned tensors that we will not move out of memory unless unpinned (tid, number of times pinned)
  std::vector<std::unordered_map<tid_t, uint32_t>> _pinned_tensors;

  // unpinned tensors, the multimap (num_copies, num_uses)
  std::vector<unpinned_t> _unpinned_tensors;

  // the gpu memory pools for fast allocation
  std::vector<gpu_memory_pool_ptr_t> _mem_pools;

  // to delete tensors
  std::vector<std::vector<tid_t>> _to_free_tensors;

  // we use these for fastpath allocations
  std::vector<std::unordered_multimap<size_t, tid_t>> _free_cache;

  // unpinned memory per gpu
  std::vector<size_t> _total_unpinned;
  
  // the total free memory per GPU
  std::vector<size_t> _total_free;

  // the total memory to free
  std::vector<size_t> _total_to_free;

  // uniquely identifies each CPU to GPU transfer
  transfer_id_t _cur_cpu_to_gpu_transfer_id = 0; 

  // the cpu to gpu transfers we have scheduled
  std::unordered_map<transfer_id_t, cpu_to_gpu_transfer_ptr_t> _cpu_to_gpu_transfer;

  // uniquely identifies each GPU to GPU transfer
  transfer_id_t _cur_gpu_tp_gpu_tranfer_id = 0;
  
  // the gpu to gpu transfers we have scheduled
  std::unordered_map<transfer_id_t, gpu_to_gpu_transfer_ptr_t> _gpu_to_gpu_transfer;

  // the number of devices
  size_t _num_devices;

  // how much memory is there per GPU
  size_t _mem_per_gpu;
};

}