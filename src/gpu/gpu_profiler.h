#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "types.h"
#include "gpu_profiler.pb.h"

namespace bbts {

class gpu_profiler_t {
public:

  gpu_profiler_t(size_t num_gpus);

  void log_cpu_copy_begin(tid_t id, size_t num_bytes, int32_t dev);
  void log_cpu_copy_end(int32_t dev);

  void log_gpu_copy_begin(int32_t dev);
  void log_gpu_copy_tensor(tid_t tid, size_t num_bytes, 
                           int32_t dst_dev, int32_t src_dev);
  void log_gpu_copy_end(int32_t dev);

  void kernel_begin(const kernel_prep_ptr_t &prep);
  void kernel_end(const kernel_prep_ptr_t &prep);

  void tensor_freed(tid_t id, int32_t dev, size_t num_bytes);
  void tensor_eviction_start(tid_t id, int32_t dev, size_t num_bytes);
  void tensor_eviction_end(tid_t id, int32_t dev);

  void log_kernel_scheduled(const kernel_prep_ptr_t &prp);
  void log_gc_scheduled(const gc_request_ptr_t &gc_req);

  void save(const std::string file_name);

private:
  
  gpu_profiler_log_t log;
};

}