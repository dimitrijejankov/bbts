#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bbts {

class block_allocator_t_expectation{
public:
  typedef size_t offset_type;
  static const offset_type invalid_offset = static_cast<offset_type>(-1);

private:
  struct free_block_info_t;

  // type of the map that keeps memory blocks sorted by their offsets
  using free_blocks_by_offset_map_t = std::map<offset_type, free_block_info_t>;

  // keep track of the expectation of different blocks; blocks are sorted on its expectation on its number of holes
  // key: expectation of holes; map is sorted by this expectation
  // value: iterator pointing to the blocks
  using expectation_to_block = std::multimap<float, offset_type>;
  

  // we need to have a hashmap that maps a list of tensor life probabilities to its corresponding block's expected number of holes
  // key: a list of probabilities, representing the probabilities of all the tensors in a block
  // value: the probability vectors for left and right block 
  using probabilities_to_expectation = std::map<std::vector<float>, float>;


  struct free_block_info_t {

    // block size (no reserved space for the size of the allocation)
    offset_type size;

    expectation_to_block::iterator order_by_expectation_it;

    // vector of all adjacent tensors expectation to stay in GPU left of this free block
    std::vector<float> left_adjacent_tensor_lives;

    // vector of all adjacent tensors expectation to stay in GPU right of this free block
    std::vector<float> right_adjacent_tensor_lives;

    free_block_info_t(offset_type _size) : size(_size) {}
  };

public:
  block_allocator_t_expectation(offset_type max_size)
      : _max_size(max_size), _free_size(max_size) {
    // insert single maximum-size block
    // add_new_block(0, _max_size);
    add_new_block_with_expectation(0, max_size, std::vector<float>(), std::vector<float>());
  }

  ~block_allocator_t_expectation() = default;

  block_allocator_t_expectation(block_allocator_t_expectation &&rhs)
      : _free_blocks_by_offset(std::move(rhs._free_blocks_by_offset)),
        _expectation_to_block(std::move(rhs._expectation_to_block)),
        _max_size(rhs._max_size), _free_size(rhs._free_size) {
    // rhs.m_max_size = 0; - const
    rhs._free_size = 0;
  }

  block_allocator_t_expectation &operator=(block_allocator_t_expectation &&rhs) = delete;
  block_allocator_t_expectation(const block_allocator_t_expectation &) = delete;
  block_allocator_t_expectation &operator=(const block_allocator_t_expectation &) = delete;

  offset_type allocate_with_expectation(offset_type size, float curr_life){
  // unlike allocate, which requires the information about blocks with empty spaces
  // we also care about information about tensor blocks as well
    assert(size != 0);
    if (_free_size < size)
      return invalid_offset;
    
    // finding the block with the lowest expectation
    // IS THIS GOING TO BE SLOW SINCE THE WORST CASE IT'S O(NUMBER OF FREE BLOCKS)?
    auto lowest_expectation_block = _expectation_to_block.begin();
    // std::cout << size << '\n';
    // std::cout << _expectation_to_block.begin()->second->second.size << '\n' << '\n';
    // assert (_expectation_to_block.begin()->second->second.size >= size);
    for (lowest_expectation_block = _expectation_to_block.begin(); lowest_expectation_block != 
              _expectation_to_block.end(); ++lowest_expectation_block){
      auto curr_block_size = _free_blocks_by_offset.find(lowest_expectation_block->second)->second.size;
      if (curr_block_size >= size){
        break;
      }
    }
    // we should have that block
    assert(_free_blocks_by_offset.count(lowest_expectation_block->second) != 0);
    assert(lowest_expectation_block !=  _expectation_to_block.end());

    auto block_by_offset = _free_blocks_by_offset.find(lowest_expectation_block->second);
    auto offset = lowest_expectation_block->second;
    auto left_lives = block_by_offset->second.left_adjacent_tensor_lives;
    auto right_lives = block_by_offset->second.right_adjacent_tensor_lives;
    
    auto new_offset = lowest_expectation_block->second + size;
    auto new_size = block_by_offset->second.size - size;
    assert(block_by_offset->second.order_by_expectation_it == lowest_expectation_block);
    _expectation_to_block.erase(lowest_expectation_block);
    _free_blocks_by_offset.erase(lowest_expectation_block->second);

    _free_size -= size;
    
    // Case 1: both left and right expectations are unspecified (i.e. this is the first time we allocate a block / the entire GPU memory is free)
    // Case 2: left expectation is unspecified; in this case we can also assign allocate our block to the leftmost area
    // Then we just arbitrarily allcate it to the leftmost area on the block
    // if (left_lives.size() == 0){
    //   // Don't need to record anything if our entire GPU memory gets occupied by this single tensor
    //   if (new_size > 0){
    //     auto new_left = std::vector<float> {curr_life};
    //     add_new_block_with_expectation(new_offset, new_size, new_left, std::vector<float>());
    //     return offset;
    //   }
    // }
    // else if (right_lives.size() == 0) {
    //   // Don't need to record anything if our entire GPU memory gets occupied by this single tensor
    //   if (new_size > 0){
    //     auto new_right = std::vector<float> {curr_life};
    //     add_new_block_with_expectation(offset, new_size, std::vector<float>(), new_right);
    //     return new_offset;
    //   }
    // }

    auto left_expectation = get_expectation(left_lives);
    auto right_expectation = get_expectation(right_lives);
    // TODO: CHECK LOGIC OF THE CODE BELOW
    // if (left_lives.size() > 0){
    //   left_life = left_lives.back();
    // }
    // if (right_lives.size() > 0){
    //   right_expectation = right_lives.front();
    // }
    
    if (right_expectation < left_expectation){
      // allocate that chuck to the rightmost of our empty block
      // we still add the new smaller block at the original offset
      //        smallest_block_it.offset
      //        |                                  |
      //        |<------smallest_block_it.size------>|
      //        |<------new_size------>|<---size---->|
      //        |                      |
      //        new block              allocated block
      if (new_size > 0){
        // since we are adding to the rightmost, we change the expectation of the right side
        std::vector<float> new_tensor_lives;
        new_tensor_lives.reserve(1 + right_lives.size());
        new_tensor_lives.push_back(curr_life);
        new_tensor_lives.insert(new_tensor_lives.end(), right_lives.begin(), right_lives.end());
        // NOTE: may need to use deque instead for more efficient performance
        add_new_block_with_expectation(offset, new_size, left_lives, new_tensor_lives);
      }
      // else need to change expectaion since we filled a hole completely between two blocks of tensors
      else{
        // |<----left block------>|<--------this block---->|<------right block------>|<----empty block (unallocated)--->|
        // unknown offset          offset                   offset + size            how to set this offset?

        // concat all expectations
        std::vector<float> new_tensor_lives;
        new_tensor_lives.reserve(left_lives.size() + 1 + right_lives.size());
        new_tensor_lives.insert(new_tensor_lives.end(), left_lives.begin(), left_lives.end());
        new_tensor_lives.insert(new_tensor_lives.end(), curr_life);
        new_tensor_lives.insert(new_tensor_lives.end(), right_lives.begin(), right_lives.end());

        auto left_block = _free_blocks_by_offset.lower_bound(offset);
        // find the free block that immediately goes after our free block
        auto right_block = _free_blocks_by_offset.upper_bound(new_offset);

        if (left_block != _free_blocks_by_offset.begin()){
          // find the free block that immediately goes before our free block
          left_block = --left_block;
          left_block->second.right_adjacent_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(left_block->first) != 0);
          _expectation_to_block.erase(left_block->second.order_by_expectation_it);
          auto left_new_expectation = std::min(get_expectation(left_block->second.left_adjacent_tensor_lives), get_expectation(left_block->second.right_adjacent_tensor_lives));
          // _expectation_to_block[left_new_expectation] = left_block;
          _expectation_to_block.emplace(left_new_expectation, left_block->first);
        }
        if (right_block != _free_blocks_by_offset.end()){
          right_block->second.left_adjacent_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(right_block->first) != 0);
          _expectation_to_block.erase(right_block->second.order_by_expectation_it);
          auto right_new_expectation = std::min(get_expectation(right_block->second.left_adjacent_tensor_lives), get_expectation(right_block->second.right_adjacent_tensor_lives));
          // _expectation_to_block[right_new_expectation] = right_block;
          _expectation_to_block.emplace(right_new_expectation, right_block->first);
        }

        // TODO discuss the case which
        // |-----left free block ----|-----pre-existing tensors---|-----current free block----|------pre-existing tensors------|----right free block---|
        // In this case if the current free block gets occupied entirely by some tensor
      }
      return (offset + new_size);
    }
    else{
      // allocate that chuck to the leftmost of our empty block
      //        smallest_block_it.offset
      //        |                                  |
      //        |<------smallest_block_it.size------>|
      //        |<------new_size------>|<---size---->|
      //        |                      |
      //        allocated block        new block
      if (new_size > 0){
        // since we are adding to the leftmost, we change the expectation of the left side
        std::vector<float> new_tensor_lives;
        new_tensor_lives.reserve(1 + left_lives.size());
        new_tensor_lives.insert(new_tensor_lives.end(), left_lives.begin(), left_lives.end());
        new_tensor_lives.push_back(curr_life);
        add_new_block_with_expectation(new_offset, new_size, new_tensor_lives, right_lives);
      }
      // else need to change expectaion since we filled a hole completely between two blocks of tensors
      else{
        // concat all expectations
        std::vector<float> new_tensor_lives;
        new_tensor_lives.reserve(left_lives.size() + 1 + right_lives.size());
        new_tensor_lives.insert(new_tensor_lives.end(), left_lives.begin(), left_lives.end());
        new_tensor_lives.insert(new_tensor_lives.end(), curr_life);
        new_tensor_lives.insert(new_tensor_lives.end(), right_lives.begin(), right_lives.end());

        auto left_block = _free_blocks_by_offset.lower_bound(offset);
        // find the free block that immediately goes after our free block
        auto right_block = _free_blocks_by_offset.upper_bound(new_offset);

        if (left_block != _free_blocks_by_offset.begin()){
          // find the free block that immediately goes before our free block
          left_block = --left_block;
          left_block->second.right_adjacent_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(left_block->first) != 0);
          _expectation_to_block.erase(left_block->second.order_by_expectation_it);
          auto left_new_expectation = std::min(get_expectation(left_block->second.left_adjacent_tensor_lives), get_expectation(left_block->second.right_adjacent_tensor_lives));
          // _expectation_to_block[left_new_expectation] = left_block;
          _expectation_to_block.emplace(left_new_expectation, left_block->first);
        }
        if (right_block != _free_blocks_by_offset.end()){
          right_block->second.left_adjacent_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(right_block->first) != 0);
          _expectation_to_block.erase(right_block->second.order_by_expectation_it);
          auto right_new_expectation = std::min(get_expectation(right_block->second.left_adjacent_tensor_lives), get_expectation(right_block->second.right_adjacent_tensor_lives));
          // _expectation_to_block[right_new_expectation] = right_block;
          _expectation_to_block.emplace(right_new_expectation, right_block->first);
        }
      }
      return offset;
    }
  }

  void free_with_expectation(offset_type offset, offset_type size, std::map<size_t, int32_t> _offset_to_tid) {
    assert(offset + size <= _max_size);

    // create new tensor life probability vectors
    std::vector<float> new_left;
    std::vector<float> new_right;

    // find the first element whose offset is greater than the specified offset.
    // upper_bound() returns an iterator pointing to the first element in the
    // container whose key is considered to go after k.
    auto next_block_it = _free_blocks_by_offset.upper_bound(offset);

    // block being deallocated must not overlap with the next block
    assert(next_block_it == _free_blocks_by_offset.end() ||
           offset + size <= next_block_it->first);
    auto prev_block_it = next_block_it;
    if (prev_block_it != _free_blocks_by_offset.begin()) {
      --prev_block_it;
      // block being deallocated must not overlap with the previous block
      assert(offset >= prev_block_it->first + prev_block_it->second.size);
    } else {
      prev_block_it = _free_blocks_by_offset.end();
    }

    offset_type new_size, new_offset;
    if (prev_block_it != _free_blocks_by_offset.end() &&
        offset == prev_block_it->first + prev_block_it->second.size) {
      //       prev_block.offset          offset
      //       |                          |
      //       |<-----prev_block.size----->|<------size-------->|
      //
      new_size = prev_block_it->second.size + size;
      new_offset = prev_block_it->first;

      if (next_block_it != _free_blocks_by_offset.end() &&
          offset + size == next_block_it->first) {
        // Case: combining current block with the left & right adjacent free block
        //     prev_block.offset          offset               next_block.offset
        //     |                          |                    |
        //     |<-----prev_block.size----->|<------size-------->|<-----next_block.size----->|
        //
        new_size += next_block_it->second.size;
        new_left = prev_block_it->second.left_adjacent_tensor_lives;
        new_right = next_block_it->second.right_adjacent_tensor_lives;
        // _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        // _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
        // Delete the range of two blocks
        ++next_block_it;
        _free_blocks_by_offset.erase(prev_block_it, next_block_it);
      } else {
        //     Case: Merging with the left free block but not the right free block
        //     prev_block.offset          offset next_block.offset
        //     |                          |                             |
        //     |<-----prev_block.size----->|<------size-------->| ~ ~ ~
        //     |<-----next_block.size----->|
        //
        new_left = prev_block_it->second.left_adjacent_tensor_lives;
        std::vector<float> new_right(next_block_it->second.left_adjacent_tensor_lives.begin() + 1, next_block_it->second.left_adjacent_tensor_lives.end());
        // _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        _free_blocks_by_offset.erase(prev_block_it);
      }
    } else if (next_block_it != _free_blocks_by_offset.end() &&
               offset + size == next_block_it->first) {
      //     Case: Merging with the right free block but not the left free block
      //     prev_block.offset                  offset next_block.offset | | |
      //     |<-----prev_block.size----->| ~ ~ ~
      //     |<------size-------->|<-----next_block.size----->|
      //
      std::vector<float> new_left(prev_block_it->second.right_adjacent_tensor_lives.begin(), prev_block_it->second.right_adjacent_tensor_lives.end() - 1);
      new_right = next_block_it->second.right_adjacent_tensor_lives;
      new_size = size + next_block_it->second.size;
      new_offset = offset;
      // _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
      _free_blocks_by_offset.erase(next_block_it);
    } else {
      // Case: the block free is in between two tensors occupying memory
      //     prev_block.offset                  offset next_block.offset
      //     |                                  |                            |
      //     |<-----prev_block.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~
      //     |<-----next_block.size----->|
      //
      new_size = size;
      new_offset = offset;

      // In this case we need to keep track of the number tensors on the left and right of our block that is being freed

      // these two expectations should record the same information
      assert (prev_block_it->second.right_adjacent_tensor_lives == next_block_it->second.left_adjacent_tensor_lives);
      // our current block should be in the map because before freeing it is a tensor in GPU memory)
      assert (_offset_to_tid.count(offset) != 0);
      auto current_tensor_it = _offset_to_tid.find(offset);
      auto count_left = 0;
      auto count_right = 0;
      for (auto expectaion_it_left = _offset_to_tid.lower_bound(prev_block_it->first); expectaion_it_left != current_tensor_it; ++expectaion_it_left){
        ++count_left;
      }
      // we want to count from current tensor 
      for (auto expectation_it_right = current_tensor_it; expectation_it_right != --_offset_to_tid.lower_bound(next_block_it->first); ++expectation_it_right){
        ++count_right;
      }

      // make sure the length add up correctly
      std::cout << count_left << '\n';
      std::cout << count_right << '\n';
      std::cout << prev_block_it->second.right_adjacent_tensor_lives.size() - 1 << '\n';
      std::cout << '\n';

      assert(count_left + count_right == prev_block_it->second.right_adjacent_tensor_lives.size() - 1);

      std::vector<float> new_left(prev_block_it->second.right_adjacent_tensor_lives.begin(), prev_block_it->second.right_adjacent_tensor_lives.begin() + count_left);
      std::vector<float> new_right(prev_block_it->second.right_adjacent_tensor_lives.begin() + count_left + 1, prev_block_it->second.right_adjacent_tensor_lives.end());
      assert(new_left.size() + new_right.size() + 1 == prev_block_it->second.right_adjacent_tensor_lives.size());
    }

    add_new_block_with_expectation(new_offset, new_size, new_left, new_right);

    _free_size += size;
  }

  offset_type get_max_size() const { return _max_size; }
  bool is_full() const { return _free_size == 0; };
  bool is_empty() const { return _free_size == _max_size; };
  offset_type get_free_size() const { return _free_size; }

private:
  void add_new_block_with_expectation(offset_type offset, offset_type size, std::vector<float> left, std::vector<float>right){
    auto new_block_it = _free_blocks_by_offset.emplace(offset, size);
    assert(new_block_it.second);
    // auto order_it = _free_blocks_by_size.emplace(size, new_block_it.first);
    auto new_expectation = std::min(get_expectation(left), get_expectation(right));
    auto expectation_it = _expectation_to_block.emplace(new_expectation, new_block_it.first->first);
    new_block_it.first->second.left_adjacent_tensor_lives = left;
    new_block_it.first->second.right_adjacent_tensor_lives = right;
    // new_block_it.first->second.order_by_size_it = order_it;
    new_block_it.first->second.order_by_expectation_it = expectation_it;
  }

  float get_expectation(std::vector<float> probabilities){
    if (probabilities.size() == 0){
      return 0;
    }

    auto exp_iter = _prob_to_exp.find(probabilities);
    if (exp_iter != _prob_to_exp.end()){
      return exp_iter->second;
    }
    
    if (probabilities.size() == 1){
      return probabilities[0];
    }
    if (probabilities.size() == 2){
      return (1-(1- probabilities[0]) * (1 - probabilities[1]));
    }
    auto new_prob = std::vector<float>(probabilities.begin() + 1, probabilities.end());
    return (1-probabilities[0]) * get_expectation(new_prob) + probabilities[0] * get_expectation_left_fixed(probabilities);
  }

  float get_expectation_left_fixed(std::vector<float> probabilities){
    auto exp_iter = _prob_to_exp_left_fixed.find(probabilities);
    if (exp_iter != _prob_to_exp_left_fixed.end()){
      return exp_iter->second;
    }

    if (probabilities.size() == 1 || probabilities.size() == 2){
      return 1;
    }
    auto new_prob_remove_left = std::vector<float>(probabilities.begin() + 1, probabilities.end());
    auto new_prob_remove_left_two = std::vector<float>(probabilities.begin() + 2, probabilities.end());
    return (probabilities[1]) * get_expectation_left_fixed(new_prob_remove_left) + (1-probabilities[1]) * (1 + get_expectation(new_prob_remove_left_two));
  }

  free_blocks_by_offset_map_t _free_blocks_by_offset;
  expectation_to_block _expectation_to_block;
  probabilities_to_expectation _prob_to_exp;
  probabilities_to_expectation _prob_to_exp_left_fixed;
  
  

  const offset_type _max_size = 0;
  offset_type _free_size = 0;
};

using block_allocator_t_expectation_ptr_t = std::shared_ptr<block_allocator_t_expectation>;

} // namespace bbts