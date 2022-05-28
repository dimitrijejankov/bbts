#pragma once

#include <cstdint>
#include <vector>
#include "../tensor/tensor.h"


namespace bbts {

enum class coordinator_op_types_t : int32_t {

  FAIL, // indicates a coordinator failure
  SCHEDULE, // batch schedules commands
  RUN, // runs all the commands
  CLEAR, // clear the storage
  SHUTDOWN, // clear the storage
  VERBOSE, // turn on or off the debug messages
  MAX_STORAGE, // maximum storage
  PRINT_STORAGE, // prints the storage of the node
  PRINT_TENSOR, // prints tensor
  REGISTER, // register a library
  FETCH_META, // fetches meta data from each node
  LOAD_TENSOR_LIST, // loads a list of tensors
  LOAD_TENSOR // loads a tensor
  // STACK_TENSOR // stacks a bunch of tensors
};

struct coordinator_op_t {

  // the type of the op
  coordinator_op_types_t _type;

  // used by schedule
  std::size_t _val;

  // used for 32 bit values like tids etc...
  std::int32_t _small_val_1;

  // used for 32 bit values like tids etc...
  std::int32_t _small_val_2;

  std::vector<bbts::tid_t> _stack_list;
};



}
