#include "reduce_op.h"
#include <cassert>
#include <cstddef>

namespace bbts {

reduce_op_t::reduce_op_t(bbts::communicator_t &_comm, bbts::tensor_factory_t &_factory,
                         bbts::storage_t &_storage, const bbts::command_t::node_list_t &_nodes,
                         command_id_t _tag, const std::vector<bbts::tid_t> &_inputs, const ud_impl_t::tensor_params_t &_params,
                         bbts::tid_t _out_tid, const bbts::ud_impl_t &_reduce_op) : _comm(_comm),
                                                                                    _factory(_factory),
                                                                                    _storage(_storage),
                                                                                    _nodes(_nodes),
                                                                                    _tag(_tag),
                                                                                    _out_tid(_out_tid),
                                                                                    _params(_params),
                                                                                    _inputs(_inputs),
                                                                                    _reduce_op(_reduce_op),
                                                                                    _input_tensors({nullptr, nullptr}),
                                                                                    _output_tensor({nullptr}),
                                                                                    _input_meta({nullptr, nullptr}),
                                                                                    _output_meta({&_out_meta}) {

  // get the impl_id of the output
  _id = _factory.get_tensor_ftm(_reduce_op.outputTypes.front());
}

int32_t reduce_op_t::get_num_nodes() const {
  return _nodes.size();
}

int32_t reduce_op_t::get_local_rank() const {

  auto it = _nodes.find(_comm.get_rank());
  return it.distance_from(_nodes.begin());
}

int32_t reduce_op_t::get_global_rank(int32_t local_rank) const {
  return _nodes[local_rank];
}

void reduce_op_t::apply() {

  int32_t mask = 0x1;
  int32_t lroot = 0;

  // relative rank
  int32_t vrank = (get_local_rank() - lroot + get_num_nodes()) % get_num_nodes();

  // apply pre-aggregation if necessary
  assert(_inputs.size() != 0);
  bbts::tid_t _in = _inputs.front();
  bbts::tid_t lhs = _inputs.front();
  bbts::tid_t rhs;

  // the output sizes and temporary size
  size_t output_size;
  size_t tmp_size;

  // do stuff
  int32_t source;
  while (mask < get_num_nodes()) {

    // receive 
    if ((mask & vrank) == 0) {
      
      source = (vrank | mask);
      if (source < get_num_nodes()) {

        // wait till we get a message from the right node
        source = (source + lroot) % get_num_nodes();

        // try to get the number of bytes to recieve
        auto rnk = get_global_rank(source);
        auto t = _comm.recv_tensor_size(rnk, _tag);

        // extract the size and success
        auto rhs_size = std::get<0>(t);
        auto success = std::get<1>(t);

        // check if there is an error
        if (!success) {
          std::cout << "Failed to recieve the tensors size for a REDUCE operation\n";
        }

        // do the recieving and calculate the output tensor size
        
        _storage.remote_transaction_p2p(_tag, rnk, {lhs}, {{TID_NONE, rhs_size}}, 
        [&](const storage_t::reservation_result_t &res) {
          
          // get the left tensor as we need it for the output
          auto l = res.get[0].get().tensor;
          
          // allocate a buffer for the tensor we are recieving
          auto r = res.create[0].get().tensor;

          // recieve the request and check if there is an error
          if (!_comm.receive_request_sync(rnk, _tag, r, rhs_size)) {
            std::cout << "Failed to recieve the tensors for a REDUCE operation\n";
          }

          // how much do we need to allocated
          _input_meta.set<0>(l->get_meta<tensor_meta_t>());
          _input_meta.set<1>(r->get_meta<tensor_meta_t>());

          // get the meta data
          _reduce_op.get_out_meta(_params, _input_meta, _output_meta);

          // set the format as get_out_meta is not responsble for doing that
          _out_meta.fmt_id = _id;

          // return the size of the tensor
          output_size = _factory.get_tensor_size(_output_meta.get<0>());

          // get the size required to 
          tmp_size = _reduce_op.get_required_memory(_params, _input_meta);

          // store the tid for later
          rhs = res.create[0].get().id;
        });

        // figure out the outputs
        const std::vector<std::tuple<tid_t, size_t>> outs = tmp_size == 0 ? std::vector<std::tuple<tid_t, size_t>>{{TID_NONE, output_size}} : 
                                                                            std::vector<std::tuple<tid_t, size_t>>{{TID_NONE, output_size}, {TID_NONE, tmp_size}};

        tid_t out_tid;
        tid_t additional_tid = TID_NONE;
        _storage.local_transaction({lhs, rhs}, outs, [&](const storage_t::reservation_result_t &res) {
        
          // get the left and right tensor so we can apply the kernel
          auto l = res.get[0].get().tensor;
          auto r = res.get[1].get().tensor;

          // allocate and init the output
          auto out = res.create[0].get().tensor;
          _factory.init_tensor(out, _out_meta);

          // get the temporary memory required
          _params.set_additional_memory(nullptr);
          if(tmp_size != 0) {
            _params.set_additional_memory(res.create[1].get().tensor->get_data_ptr<void*>());
            additional_tid = res.create[1].get().id;
          }

          // store the tid for later
          out_tid = res.create[0].get().id;

          // set the input tensors to the function
          _input_tensors.set<0>(*l);
          _input_tensors.set<1>(*r);

          // set the output tensor to the function
          _output_tensor.set<0>(*out);

          // run the function
          _reduce_op.call_ud(_params, _input_tensors, _output_tensor);
        });

        // remove the additional memory
        if(additional_tid != TID_NONE) {
          _storage.remove_by_tid(additional_tid);
        }

        // manage the memory
        if(lhs != _in) {
            _storage.remove_by_tid(lhs);
        }
        _storage.remove_by_tid(rhs);
        
        // set the lhs
        lhs = out_tid;
      }

    } else {

      // I've received all that I'm going to.  Send my result to my parent
      source = ((vrank & (~mask)) + lroot) % get_num_nodes();

      // caucluate the global rank
      auto rnk = get_global_rank(source);

      // get the size of the tensor
      auto num_bytes = _storage.get_tensor_size(lhs);
      
      // send the tensor size and check if there is any error
      if(!_comm.send_tensor_size(rnk, _tag, num_bytes)) {
        std::cout << "Communication failure, could not send the tensor size while REDUCING.\n";
        exit(-1);
      }

      // init a transaction to send the tensor
      _storage.remote_transaction_p2p(_tag, rnk, {lhs}, {}, 
      [&](const storage_t::reservation_result_t &res) {

        // send the tensor synchronously
        auto l  = res.get[0].get().tensor;
        if (!_comm.send_sync(l, num_bytes, rnk, _tag)) {        
            std::cout << "Communication failure, could not send the tensor size while REDUCING.\n";
        }

      });

      break;
    }
    mask <<= 1;
  }

  // free the lhs
  if(get_local_rank() != 0) {
    if(lhs != _in) {
      _storage.remove_by_tid(lhs);
    }
  }
  else {

    // assign a tid to the result of the aggregation
    _storage.assign_tid(lhs, _out_tid);
  }
}

}