#include "types.h"

namespace bbts {

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

}