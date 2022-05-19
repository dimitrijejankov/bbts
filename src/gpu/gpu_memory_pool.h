#include "types.h"
#include "../storage/block_allocator.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace bbts {

class gpu_memory_pool_t {
public:
  gpu_memory_pool_t(int32_t dev, size_t num_bytes) : block_allocator(num_bytes) {

    cudaSetDevice(dev);

    // allocate the memory
    checkCudaErrors(cudaMalloc(&memory, num_bytes));
  }

  ~gpu_memory_pool_t() {

    // free the memory
    cudaFree(memory);
  }

  void* allocate(size_t num_bytes) {

    // allocate the host pinned memory
    auto offset = block_allocator.allocate(num_bytes);
    if(offset == block_allocator.invalid_offset) {
      throw std::runtime_error("Failed to allocate!");
    }
    return memory + block_allocator.allocate(num_bytes);
  }

  void free(void *mem, size_t num_bytes) {
    
    // free the GPU
    size_t offset = (size_t)((uint8_t*) mem - memory);
    block_allocator.free(offset, num_bytes);
  }

  uint8_t *memory;

  block_allocator_t block_allocator;
};

using gpu_memory_pool_ptr_t = std::shared_ptr<gpu_memory_pool_t>;

}