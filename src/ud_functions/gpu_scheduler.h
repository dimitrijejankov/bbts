#include "../server/static_config.h"
#include "gpu_scheduler_impl.h"
#include "null_gpu_scheduler.h"

#include <cstddef>
#include <memory>

namespace bbts {

#ifdef ENABLE_GPU
using gpu_scheduler_t = gpu_scheduler_impl_t;
#else
using gpu_scheduler_t = null_gpu_scheduler_t;
#endif

using gpu_scheduler_ptr_t = std::shared_ptr<gpu_scheduler_t>;

} // namespace bbts