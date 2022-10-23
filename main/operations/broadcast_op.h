#pragma once

#include "../communication/communicator.h"
#include "../storage/storage.h"
#include "../tensor/builtin_formats.h"
#include "../ud_functions/udf_manager.h"

#include <cstddef>
#include <iostream>
#include <algorithm>
#include <mpi.h>

namespace bbts {

// implementation based on the openmpi implementation
class broadcast_op_t {
public:

  // constructs the broadcast operation, the root node is assumed to be the first in the _nodes array
  broadcast_op_t(bbts::communicator_t &_comm,
                 bbts::storage_t &_storage,
                 const bbts::command_t::node_list_t &_nodes,
                 int32_t _tag, 
                 size_t _num_bytes, 
                 bbts::tid_t _tid);

  // the mpi communicator we are going to use to perform the communication
  bbts::communicator_t &_comm;

  // the storage we use this to allocate the output and the intermediate tensors
  bbts::storage_t &_storage;

  // the nodes
  bbts::command_t::node_list_t _nodes;

  // the tag we use to identify this reduce
  int32_t _tag;

  // the input tensor of this node
  bbts::tensor_t *_in;

  // the number of bytes
  size_t _num_bytes;

  // the tid
  bbts::tid_t _tid;

  // calculates the highest bit in an integer
  static inline int opal_hibit(int value, int start) {
    unsigned int mask;

    /* Only look at the part that the caller wanted looking at */
    mask = value & ((1 << start) - 1);

    if ((0 == mask)) {
      [[unlikely]]
      return -1;
    }

    start = (8 * sizeof(int) - 1) - __builtin_clz(mask);

    return start;
  }

  // cubedim The smallest cube dimension containing that value
  static inline int opal_cube_dim(int value) {
    int dim, size;

    if ((1 >= value)) {
      [[unlikely]]
      return 0;
    }
    size = 8 * sizeof(int);
    dim = size - __builtin_clz(value - 1);

    return dim;
  }

  // returns the number of nodes used for the broadcast
  [[nodiscard]] int32_t get_num_nodes() const;

  // returns the local rank within the broadcast
  [[nodiscard]] int32_t get_local_rank() const;

  // returns the global rank
  [[nodiscard]] int32_t get_global_rank(int32_t local_rank) const;

  // runs the broadcast
  void apply();
};

}