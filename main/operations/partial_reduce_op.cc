#include "partial_reduce_op.h"

bbts::partial_reduce_op_t::partial_reduce_op_t(bbts::tensor_factory_t &factory, 
                                               bbts::storage_t &storage,
                                               const tid_t lhs,
                                               const tid_t rhs,
                                               tid_t &out_tid,
                                               ud_impl_t::tensor_params_t &params,
                                               const bbts::ud_impl_t &reduce_op) : 
                                                                  _factory(factory),
                                                                  _storage(storage),
                                                                  _lhs(lhs),
                                                                  _rhs(rhs),
                                                                  _params(params),
                                                                  _reduce_op(reduce_op),
                                                                  _out_tid(out_tid),
                                                                  _input_tensors({nullptr, nullptr}),
                                                                  _output_tensor({nullptr}),
                                                                  _input_meta({nullptr, nullptr}),
                                                                  _output_meta({&_out_meta})  {

  // get the impl_id of the output
  _id = factory.get_tensor_ftm(reduce_op.outputTypes.front());
}

void bbts::partial_reduce_op_t::apply() {

  // calculate the size of the output tensor
  size_t output_size;
  size_t tmp_size = 0;
  _storage.local_transaction({_lhs, _rhs}, {}, [&](const storage_t::reservation_result_t &res) {

    auto l = res.get[0].get();
    auto r = res.get[1].get();

    // how much do we need to allocated
    _input_meta.set<0>(l.tensor->as<bbts::tensor_meta_t>());
    _input_meta.set<1>(r.tensor->as<bbts::tensor_meta_t>());

    // get the meta data
    _reduce_op.get_out_meta(_params, _input_meta, _output_meta);

    // set the output meta
    auto &type = _reduce_op.outputTypes[0];
    _output_meta.get_by_idx(0).fmt_id = _factory.get_tensor_ftm(type);

    // return the size of the tensor
    output_size = _factory.get_tensor_size(_output_meta.get<0>());

    // get the size required to 
    tmp_size = _reduce_op.get_required_memory(_params, _input_meta);
  });

  // figure out if this is the an intermediate reduce
  auto to_create = _out_tid < 0 ? TID_NONE : _out_tid;

  // perform the actual kernel
  tid_t additional_tid = TID_NONE;
  _storage.local_transaction({_lhs, _rhs}, {{to_create, output_size}, {TID_NONE, tmp_size}}, [&](const storage_t::reservation_result_t &res) {
  
    // init the output tensor
    auto &out = res.create[0].get().tensor;
    _factory.init_tensor(out, _out_meta);

    // get the left and right tensor
    auto l = res.get[0].get().tensor;
    auto r = res.get[1].get().tensor;

    // set the input tensors to the function
    _input_tensors.set<0>(*l);
    _input_tensors.set<1>(*r);

    // set the output tensor to the function
    _output_tensor.set<0>(*out);

    // get the temporary memory required
    _params.set_additional_memory(nullptr);
    if(tmp_size != 0) {
      _params.set_additional_memory(res.create[1].get().tensor->get_data_ptr<void*>());
      additional_tid = res.create[1].get().id;
    }

    // run the function
    _reduce_op.call_ud(_params, _input_tensors, _output_tensor);

    // set the output tid
    _out_tid = res.create[0].get().id;
  });

  // remove the additional memory
  if(additional_tid != TID_NONE) {
    _storage.remove_by_tid(additional_tid);
  }
}