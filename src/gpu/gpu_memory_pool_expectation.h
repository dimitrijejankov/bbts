#include "types.h"
#include "../storage/block_allocator_expectation.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace bbts {

class gpu_memory_pool_t_expectation {

  // at the highest user level application, given user specified operations, the system should generate a dataflow graph based on logical constraints
  // from the dataflow graph, we should be able to get estimate for how long each of the tensor will stay in the system during the application
  // NOTE: the current estimate is very naive
  // the lowest level allocator (block allocator) need to keep track of this info to make smarter allocation decisions
  using tid_to_probability = std::map<size_t, float>;

public:
  gpu_memory_pool_t_expectation(int32_t dev, size_t num_bytes) : block_allocator_t_expectation(num_bytes) {

    // allocate the memory
    this->dev = dev;
    cudaSetDevice(dev);
    checkCudaErrors(cudaMalloc(&memory, num_bytes));
  }

  ~gpu_memory_pool_t_expectation() {

    // free the memory
    cudaFree(memory);
  }

  void* allocate_with_expectation(size_t num_bytes, float tensor_life_prob, std::set<size_t> &_tensor_offsets, tid_t tid, int32_t dev) {

    // allocate the host pinned memory
    ++allocation_count;
    // TODO: do more check on this 
    // assert(_tensor_lifemap.count(tid) != 0);
    block_allocator_t::offset_type real_offset;
    // if (_tensor_lifemap.count(tid) != 0){
    //   real_offset = block_allocator.allocate_with_expectation(num_bytes + target_aligment, tensor_life_prob);
    // }
    // // we got a negative tensor (tensor that act as a immediate)
    // else if (tid < 0){
    //   real_offset = block_allocator.allocate_with_expectation(num_bytes + target_aligment, .0001);
    // }
    // else{
    //   real_offset = block_allocator.allocate_with_expectation(num_bytes + target_aligment, .0001);
    // }
    real_offset = block_allocator_t_expectation.allocate_with_expectation(num_bytes + target_aligment, tensor_life_prob);
    if(real_offset == block_allocator_t_expectation.invalid_offset) {
      return nullptr;
    }

    // align the address
    size_t to_offset = real_offset % target_aligment != 0 ? target_aligment - (real_offset % target_aligment) : 0;
    auto aligned_offset = real_offset + to_offset;

    // store the offsets
    offset_to_aligned_offset.insert({real_offset, aligned_offset});
    aligned_offset_to_offset.insert({aligned_offset, real_offset});

    _tensor_offsets.insert(real_offset);

    std::cout << "Dev: " << dev << " Allocating Block: Size: " << num_bytes << " Allocated Offset: " 
                  << real_offset << " tid: " << tid << " # of total allocates: " << allocation_count << "\n";

    return memory + aligned_offset;
  }

  void free_with_offset_map(void *aligned_address, size_t num_bytes, std::set<size_t> &_tensor_offsets, int32_t dev) {

    ++free_count;

    // free the GPU
    auto aligned_offset = (size_t) ((uint8_t*) aligned_address - memory);
    auto real_offset = aligned_offset_to_offset[aligned_offset];

    // just free this
    block_allocator_t_expectation.free_with_expectation(real_offset, num_bytes + target_aligment, _tensor_offsets);

    offset_to_aligned_offset.erase(real_offset);
    aligned_offset_to_offset.erase(aligned_offset);

    _tensor_offsets.erase(real_offset);

    std::cout << "Dev: " << dev << " Freed tensor from offset: " << real_offset << " # of total frees: " << free_count << "\n";
  }

  size_t get_real_offset(void *aligned_address){
    auto aligned_offset = (size_t) ((uint8_t*) aligned_address - memory);
    return aligned_offset_to_offset[aligned_offset];
  }


  // offset to aligned offset
  std::unordered_map<size_t, size_t> offset_to_aligned_offset;
  std::unordered_map<size_t, size_t> aligned_offset_to_offset;

  const size_t target_aligment = 256;

  uint8_t *memory;

  int32_t dev;

  block_allocator_t_expectation block_allocator_t_expectation;

  tid_to_probability _tensor_lifemap;

  int32_t allocation_count;
  int32_t free_count;
};

using gpu_memory_pool_t_expectation_ptr_t = std::shared_ptr<gpu_memory_pool_t_expectation>;

}