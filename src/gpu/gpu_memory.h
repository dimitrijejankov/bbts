#include "types.h"
#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <unordered_map>

namespace bbts {

// the gpu memory take care of the following three things:
// 1. it does preallocation of the GPU memory
// 2. it issues garbage collection requests in the case that we need to free some memory
// 3. it issues transfers to the kernel preps
class gpu_memory_t {
public:

  gpu_memory_t(size_t num_devixces, size_t mem_per_gpu);

  // mark that we are using this tensor in the future
  void mark_for_use(tid_t id);

  // mark that a tensor has already been used
  void mark_as_used(tid_t id);

  // mark this tensor for deletion
  void mark_for_deletion(tid_t id);

  // pins all the tensors in the kernel prep
  void pin_all(kernel_prep_ptr_t kp, int dev);

  // unpins all the tensors in the kernel prep
  void unpin_all(kernel_prep_ptr_t kp, int dev);
  
  // mark that a tensor is not loaded and on a particular device
  void tensor_created(tid_t id, int dev);

  // can we preallocate all the tensors that we need to run the kernel
  int can_preallocate(kernel_prep_ptr_t kp);

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

  // can an we run garbage collection
  // if we can it returns the device othewise it returns -1
  int can_gc(kernel_prep_ptr_t kp);

  // get the garbage collection request
  gc_request_ptr_t get_gc_request(kernel_prep_ptr_t kp, int dev);

private:
  
  // check whether the tensor is present on the device (either pinned or unpinned)
  bool _is_on_device(tid_t id, int32_t dev);

  void _pin_tensor(tid_t id, int32_t dev, size_t num_bytes);

  // mark that a tensor is not unpinned
  void _unpin_tensor(tid_t id, int dev, size_t num_bytes);

  // sorted by num_copies (first), num_uses (second)
  using unpinned_t = std::multimap<std::tuple<uint32_t, uint32_t>, tid_t>;

  struct gpu_mem_tensor_t {
    
    // the number of uses of this tensor
    uint32_t num_uses = 0;

    // how many copies of this tensor are there in memory
    uint32_t num_copies = 0;

    // these are iterators so that we can quickly update the _unpinned_tensors
    std::array<unpinned_t::iterator, BBTS_MAX_GPU_DEVICES> unpinned_its;
  };

  // how many times will this tensor be used
  std::unordered_map<tid_t, gpu_mem_tensor_t> _tensors;

  // these are the pinned tensors that we will not move out of memory unless unpinned (tid, number of times pinned)
  std::vector<std::unordered_map<tid_t, uint32_t>> _pinned_tensors;

  // unpinned tensors, the multimap (num_copies, num_uses)
  std::vector<unpinned_t> _unpinned_tensors;

  // to delete tensors
  std::vector<std::vector<tid_t>> _to_free_tensors;

  // unpinned memory per gpu
  std::vector<size_t> _total_unpinned;
  
  // the total free memory per GPU
  std::vector<size_t> _total_free;

  size_t _num_devices;
};

}