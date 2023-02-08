#include "types.h"
#include "../storage/block_allocator.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace bbts {

class gpu_memory_pool_t {
public:
  gpu_memory_pool_t(int32_t dev, size_t num_bytes) : block_allocator(num_bytes) {

    // allocate the memory
    this->dev = dev;
    cudaSetDevice(dev);
    checkCudaErrors(cudaMalloc(&memory, num_bytes));
  }

  ~gpu_memory_pool_t() {

    // free the memory
    cudaFree(memory);
  }

  void* allocate(size_t num_bytes) {

    ++allocation_count;

    // allocate the host pinned memory
    auto real_offset = block_allocator.allocate(num_bytes + target_aligment);
    if(real_offset == block_allocator.invalid_offset) {
      return nullptr;
    }

    // align the address
    size_t to_offset = real_offset % target_aligment != 0 ? target_aligment - (real_offset % target_aligment) : 0;
    auto aligned_offset = real_offset + to_offset;

    // store the offsets
    offset_to_aligned_offset.insert({real_offset, aligned_offset});
    aligned_offset_to_offset.insert({aligned_offset, real_offset});

    std::cout << "Original Dev: " << dev << " Allocating Block: Size: " << num_bytes << " Allocated Offset: " 
                  << real_offset << " # of total allocates: " << allocation_count << "\n";

    return memory + aligned_offset;
  }

  void free(void *aligned_address, size_t num_bytes) {

    ++free_count;

    // free the GPU
    auto aligned_offset = (size_t) ((uint8_t*) aligned_address - memory);
    auto real_offset = aligned_offset_to_offset[aligned_offset];

    // just free this
    block_allocator.free(real_offset, num_bytes + target_aligment);

    offset_to_aligned_offset.erase(real_offset);
    aligned_offset_to_offset.erase(aligned_offset);

    std::cout << "Original Dev: " << dev << " Freed tensor from offset: " << real_offset << " # of total frees: " << free_count << "\n";
  }

  // offset to aligned offset
  std::unordered_map<size_t, size_t> offset_to_aligned_offset;
  std::unordered_map<size_t, size_t> aligned_offset_to_offset;

  const size_t target_aligment = 256;

  uint8_t *memory;

  int32_t dev;

  block_allocator_t block_allocator;

  int32_t allocation_count;
  int32_t free_count;
};

using gpu_memory_pool_ptr_t = std::shared_ptr<gpu_memory_pool_t>;

}