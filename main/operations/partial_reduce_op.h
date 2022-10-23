#pragma once

#include "../communication/communicator.h"
#include "../tensor/builtin_formats.h"
#include "../storage/storage.h"
#include "../ud_functions/udf_manager.h"
#include <iostream>
#include <algorithm>

namespace bbts {

class partial_reduce_op_t {
public:

  partial_reduce_op_t(bbts::tensor_factory_t &factory, 
                      bbts::storage_t &storage,
                      const tid_t lhs,
                      const tid_t rhs,
                      tid_t &out_tid,
                      const ud_impl_t::tensor_params_t &params,
                      const bbts::ud_impl_t &reduce_op);

  // apply this operation
  void apply();

  bbts::tensor_factory_t &_factory;

  bbts::storage_t &_storage;

  tid_t _lhs;

  tid_t _rhs;

  tid_t &_out_tid;

  const ud_impl_t::tensor_params_t &_params;

  const bbts::ud_impl_t &_reduce_op;

  // make empty input arguments
  bbts::tensor_meta_t _out_meta{};
  bbts::ud_impl_t::tensor_args_t _input_tensors;
  bbts::ud_impl_t::tensor_args_t _output_tensor;
  bbts::ud_impl_t::meta_args_t _input_meta;
  bbts::ud_impl_t::meta_args_t _output_meta;

  // the id of the tensor format
  bbts::tfid_t _id;
};

}