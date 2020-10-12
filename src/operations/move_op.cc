#include "move_op.h"

namespace bbts {

  // constructs the reduce operation
  move_op::move_op(bbts::communicator_t &_comm, int32_t _tag, bbts::tensor_t *_tensor,  
          bool _is_sender, bbts::tensor_factory_t &_factory, 
          bbts::storage_t &_storage, bbts::node_id_t _node) : _comm(_comm),
                                                              _tag(_tag),
                                                              _tensor(_tensor),
                                                              _is_sender(_is_sender),
                                                              _factory(_factory),
                                                              _storage(_storage),
                                                              _node(_node) {}

  // apply this operation
  bbts::tensor_t *move_op::apply() {
    
    // is this the sender, if so we initiate a send request
    if(_is_sender) {

      // get the number of bytes we need to send
      auto num_bytes = _factory.get_tensor_size(_tensor->_meta);

      // do the sending
      if(!_comm.send_sync(_tensor, num_bytes, _node, _tag)) {
        std::cout << "Error 1\n";
      }

    } else {

      // try to get the request
      auto req = _comm.expect_request_sync(_node, _tag);

      // check if there is an error
      if (!req.success) {
        std::cout << "Error 2\n";
      }

      // allocate a buffer for the tensor
      _tensor = _storage.create_tensor(req.num_bytes);

      // recieve the request and check if there is an error
      if (!_comm.recieve_request_sync(_tensor, req)) {
        std::cout << "Error 3\n";
      }
    }

    // return the tensor
    return _tensor;
  }

}