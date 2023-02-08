#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>
#include <ctime>
#include <chrono>

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

  // type of the map that keeps memory blocks sorted by their sizes
  using free_blocks_by_size_t =
      std::multimap<offset_type, offset_type>;
  

  // we need to have a hashmap that maps a list of tensor life probabilities to its corresponding block's expected number of holes
  // key: a list of probabilities, representing the probabilities of all the tensors in a block
  // value: the probability vectors for left and right block 
  using probabilities_to_expectation = std::map<std::vector<float>, float>;

  struct free_block_info_t {

    // block size (no reserved space for the size of the allocation)
    offset_type size;

    free_blocks_by_size_t::iterator order_by_size_it;

    // vector of all adjacent tensors expectation to stay in GPU left of this free block
    std::vector<float> left_tensor_lives;

    // vector of all adjacent tensors expectation to stay in GPU right of this free block
    std::vector<float> right_tensor_lives;

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
        _free_blocks_by_size(std::move(rhs._free_blocks_by_size)),
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
    
    auto lowest_expectation_block = _free_blocks_by_size.lower_bound(size);
    if (lowest_expectation_block == _free_blocks_by_size.end())
      return invalid_offset;

    size_t lowest_exp_diff = INT_MAX;
    // |-----free block-----|
    // |--0--|---------|--1-| (either 0 or 1)
    // 0: we should allocate tensor left of our free block;
    // 1: we should allocate tensor right of our free block;
    size_t left_or_right = -1;

    // auto t1 = std::chrono::high_resolution_clock::now();
    for (auto curr_block_it = lowest_expectation_block; curr_block_it != _free_blocks_by_size.end(); ++curr_block_it){
      auto block_by_offset_it = _free_blocks_by_offset.find(curr_block_it->second);
      auto left_tensors = block_by_offset_it->second.left_tensor_lives;
      auto right_tensors  = block_by_offset_it->second.right_tensor_lives;

      auto old_left_exp = get_expectation(left_tensors, true);
      auto old_right_exp = get_expectation(right_tensors, true);

      left_tensors.push_back(curr_life);
      auto new_left_exp = get_expectation(left_tensors, true);
      left_tensors.pop_back();

      if (new_left_exp - old_left_exp < lowest_exp_diff){
        lowest_expectation_block = curr_block_it;
        lowest_exp_diff = new_left_exp - old_left_exp;
        left_or_right = 0;
      }

      right_tensors.insert(right_tensors.begin(), curr_life);
      auto new_right_exp = get_expectation(right_tensors, true);
      right_tensors.erase(right_tensors.begin());

      if (new_right_exp - old_right_exp < lowest_exp_diff){
        lowest_expectation_block = curr_block_it;
        lowest_exp_diff = new_right_exp - old_right_exp;
        left_or_right = 1;
      }
    }
    // auto t2 = std::chrono::high_resolution_clock::now();
    // auto ms_int = duration_cast<std::chrono::milliseconds>(t2 - t1);
    // std::cout << "Finding the correct spot time: " << ms_int.count() << "ms\n";

    assert(left_or_right != -1);

    // we should have that block
    assert(_free_blocks_by_offset.count(lowest_expectation_block->second) != 0);
    assert(lowest_expectation_block !=  _free_blocks_by_size.end());

    auto block_by_offset = _free_blocks_by_offset.find(lowest_expectation_block->second);
    auto offset = lowest_expectation_block->second;
    auto left_lives = block_by_offset->second.left_tensor_lives;
    auto right_lives = block_by_offset->second.right_tensor_lives;
    
    auto new_offset = lowest_expectation_block->second + size;
    auto new_size = block_by_offset->second.size - size;
    assert(block_by_offset->second.order_by_size_it == lowest_expectation_block);
    _free_blocks_by_size.erase(lowest_expectation_block);
    _free_blocks_by_offset.erase(lowest_expectation_block->second);

    _free_size -= size;
    

    auto left_expectation = get_expectation(left_lives, true);
    auto right_expectation = get_expectation(right_lives, true);
    
    if (left_or_right == 1){
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
        auto right_block = _free_blocks_by_offset.upper_bound(new_offset);
        if (right_block != _free_blocks_by_offset.end()){
          right_block->second.left_tensor_lives = new_tensor_lives;
        }
      }
      // else need to change expectation since we filled a hole completely between two blocks of tensors
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
          left_block->second.right_tensor_lives = new_tensor_lives;
          assert (_free_blocks_by_offset.count(left_block->first) != 0);
        }
        if (right_block != _free_blocks_by_offset.end()){
          right_block->second.left_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(right_block->first) != 0);
        }
      }
      // std::cout << "Allocating Block: Size: " << size << " Allocated Offset: " << (offset + new_size) << "\n";
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
        auto left_block = _free_blocks_by_offset.lower_bound(offset);
        if (left_block != _free_blocks_by_offset.begin()){
          left_block = --left_block;
          left_block->second.right_tensor_lives = new_tensor_lives;
        }
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
          left_block->second.right_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(left_block->first) != 0);
        }
        if (right_block != _free_blocks_by_offset.end()){
          right_block->second.left_tensor_lives = new_tensor_lives;
          // need to change the entry in expectation map
          assert (_free_blocks_by_offset.count(right_block->first) != 0);
        }
      }
      return offset;
    }
  }

  void free_with_expectation(offset_type offset, offset_type size, std::set<size_t> _tensor_offsets) {
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
      // Check if the data block we want to free can be merged with the left free block
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
        new_left = prev_block_it->second.left_tensor_lives;
        new_right = next_block_it->second.right_tensor_lives;
        _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
        // Delete the range of two blocks
        ++next_block_it;
        _free_blocks_by_offset.erase(prev_block_it, next_block_it);
      } else {
        //     Case: Merging with the left free block but not the right free block
        //     prev_block.offset           offset                       next_block.offset
        //     |                           |                            |
        //     |<-----prev_block.size----->|<------size-------->| ~ ~ ~ |<-----next_block.size----->|
        //
        new_left = prev_block_it->second.left_tensor_lives;
        new_right= std::vector<float>(next_block_it->second.left_tensor_lives.begin() + 1, 
                                      next_block_it->second.left_tensor_lives.end());
        _free_blocks_by_size.erase(prev_block_it->second.order_by_size_it);
        _free_blocks_by_offset.erase(prev_block_it);
        // next free block's info should also be updated
        if (next_block_it != _free_blocks_by_offset.end()){
          next_block_it->second.left_tensor_lives = new_right;
        }
      }
    } else if (next_block_it != _free_blocks_by_offset.end() &&
               offset + size == next_block_it->first) {
      //     Case: Merging with the right free block but not the left free block
      //     prev_block.offset                  offset               next_block.offset 
      //     |                                  |                    |
      //     |<-----prev_block.size----->| ~ ~ ~|<------size-------->|<-----next_block.size----->|
      //
      new_left = std::vector<float>(next_block_it->second.left_tensor_lives.begin(), 
                                  next_block_it->second.left_tensor_lives.end() - 1);
      new_right = next_block_it->second.right_tensor_lives;
      new_size = size + next_block_it->second.size;
      new_offset = offset;
      _free_blocks_by_size.erase(next_block_it->second.order_by_size_it);
      _free_blocks_by_offset.erase(next_block_it);
      // prev free block's info should also be updated
      if (prev_block_it != _free_blocks_by_offset.end()){
        prev_block_it->second.right_tensor_lives = new_left;
      }
    } else {
      // Case: the block free is in between two tensors occupying memory
      //     prev_block.offset                   offset                      next_block.offset
      //     |                                   |                           |
      //     |<-----prev_block.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~|<-----next_block.size----->|
      // Note: prev_block and next_block may not exist
      new_size = size;
      new_offset = offset;

      // In this case we need to keep track of the number tensors on the left and right of our block that is being freed

      // these two expectations should record the same information
      if (prev_block_it != _free_blocks_by_offset.end() && next_block_it != _free_blocks_by_offset.end()){
        assert (prev_block_it->second.right_tensor_lives == next_block_it->second.left_tensor_lives);
      }
      
      // our current block should be in the map because before freeing it is a tensor in GPU memory)
      assert (_tensor_offsets.count(offset) != 0);
      auto current_tensor_it = _tensor_offsets.find(offset);
      auto count_left = 0;
      auto count_right = 0;
      if (prev_block_it != _free_blocks_by_offset.end()){
        // count how many tensors are between previous empty block and the block we are trying to free
        for (auto expectaion_it_left = _tensor_offsets.lower_bound(prev_block_it->first); 
                                        expectaion_it_left != current_tensor_it; ++expectaion_it_left){
          ++count_left;
        }
      }
      else{
        for (auto expectaion_it_left = _tensor_offsets.begin(); expectaion_it_left != current_tensor_it; ++expectaion_it_left){
          ++count_left;
        }
      }
      if (next_block_it != _free_blocks_by_offset.end()){
        for (auto expectation_it_right = std::next(current_tensor_it);
                                expectation_it_right != _tensor_offsets.lower_bound(next_block_it->first); ++expectation_it_right){
          ++count_right;
        }
      }
      else{
        for (auto expectation_it_right = std::next(current_tensor_it);
                                expectation_it_right != _tensor_offsets.end(); ++expectation_it_right){
          ++count_right;
        }
      }

      if (prev_block_it != _free_blocks_by_offset.end()){
        //     |<-----prev_block.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~|GPU Mem Ends
        // Or  |<-----prev_block.size----->| ~ ~ ~ |<------size-------->| ~ ~ ~|<-----next_block.size----->|
        assert(count_left + count_right == prev_block_it->second.right_tensor_lives.size() - 1);
        new_left = std::vector<float>(prev_block_it->second.right_tensor_lives.begin(), 
                                    prev_block_it->second.right_tensor_lives.begin() + count_left);
        new_right = std::vector<float>(prev_block_it->second.right_tensor_lives.begin() + count_left + 1, 
                                    prev_block_it->second.right_tensor_lives.end());

        // Do another check to make sure we don't mess up anything
        assert(new_left.size() + new_right.size() + 1 == prev_block_it->second.right_tensor_lives.size());
        // prev_block_it->second.right_adjacent_tensor_lives = new_left;
        // if (next_block_it != _free_blocks_by_offset.end()){
        //   next_block_it->second.left_adjacent_tensor_lives = new_right;
        // }
      }
      else if (next_block_it != _free_blocks_by_offset.end()){
        //  GPU MEM Beginnging| ~ ~ ~ |<------size-------->| ~ ~ ~|<-----next_block.size----->|
        assert(count_left + count_right == next_block_it->second.left_tensor_lives.size() - 1);
        new_left = std::vector<float>(next_block_it->second.left_tensor_lives.begin(), 
                                    next_block_it->second.left_tensor_lives.begin() + count_left);
        new_right = std::vector<float>(next_block_it->second.left_tensor_lives.begin() + count_left + 1, 
                                      next_block_it->second.left_tensor_lives.end());
        // Do another check to make sure we don't mess up anything
        assert(new_left.size() + new_right.size() + 1 == next_block_it->second.left_tensor_lives.size());
        // next_block_it->second.left_adjacent_tensor_lives = new_right;
      }
      else{
        std::cout << "#####The entire GPU mem is full, which is almost impossible to happen\n";
        assert(count_left + count_right == _tensor_offsets.size() - 1);
      }

      if (prev_block_it != _free_blocks_by_offset.end()){
        prev_block_it->second.right_tensor_lives = new_left;
      }
      if (next_block_it != _free_blocks_by_offset.end()){
        next_block_it->second.left_tensor_lives = new_right;
      }

      if ((new_left.size() == 0 && offset != 0)|| new_right.size() == 0){
        std::cout << "size 0 warning";
      }
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
    auto order_it = _free_blocks_by_size.emplace(size, new_block_it.first->first);
    new_block_it.first->second.left_tensor_lives = left;
    new_block_it.first->second.right_tensor_lives = right;
    // new_block_it.first->second.order_by_size_it = order_it;
    new_block_it.first->second.order_by_size_it = order_it;
  }

 float get_expectation(std::vector<float> probabilities, bool save){
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
    auto my_expectation = (1-probabilities[0]) * get_expectation(new_prob, save) 
                          + probabilities[0] * get_expectation_left_fixed(probabilities, save);
    if (save){
      _prob_to_exp.emplace(probabilities, my_expectation);
    }
    return my_expectation;
  }

  float get_expectation_left_fixed(std::vector<float> probabilities, bool save){
    auto exp_iter = _prob_to_exp_left_fixed.find(probabilities);
    if (exp_iter != _prob_to_exp_left_fixed.end()){
      return exp_iter->second;
    }

    if (probabilities.size() == 1 || probabilities.size() == 2){
      return 1;
    }
    auto new_prob_remove_left = std::vector<float>(probabilities.begin() + 1, probabilities.end());
    auto new_prob_remove_left_two = std::vector<float>(probabilities.begin() + 2, probabilities.end());
    auto my_expectation = (probabilities[1]) * get_expectation_left_fixed(new_prob_remove_left, save) 
                            + (1-probabilities[1]) * (1 + get_expectation(new_prob_remove_left_two, save));
    if (save){
      _prob_to_exp_left_fixed.emplace(probabilities, my_expectation);
    }
    return my_expectation;
  }

  free_blocks_by_offset_map_t _free_blocks_by_offset;
  free_blocks_by_size_t _free_blocks_by_size;
  probabilities_to_expectation _prob_to_exp;
  probabilities_to_expectation _prob_to_exp_left_fixed;
  
  

  const offset_type _max_size = 0;
  offset_type _free_size = 0;
};

using block_allocator_t_expectation_ptr_t = std::shared_ptr<block_allocator_t_expectation>;

} // namespace bbts