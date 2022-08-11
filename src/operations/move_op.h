#include "../communication/communicator.h"
#include "../tensor/builtin_formats.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <cstddef>
#include <mpi.h>
#include <iostream>
#include <algorithm>

namespace bbts {

class move_op_t {
public:

  // the mpi communicator we are going to use to perform the communication
  bbts::communicator_t &_comm;

  // the tag we use to identify this reduce
  command_id_t _cmd_id; 

  // the number of bytes
  size_t _num_bytes;

  // the id of the tensor we are moving
  tid_t _tid;

  // is this node the sender
  bool _is_sender;

  // we use this to allocate the tensors
  bbts::storage_t &_storage;

  // the node we are send or recieving from...
  bbts::node_id_t _node;

  // constructs the reduce operation
  move_op_t(bbts::communicator_t &_comm, command_id_t _cmd_id, 
            size_t _num_bytes, tid_t _tid, 
            bool _is_sender, bbts::storage_t &_storage, bbts::node_id_t _node);

  // apply this operation
  void apply();

};

}