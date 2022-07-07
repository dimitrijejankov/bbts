#pragma once

#include <cassert>
#include <cstdint>
#include "ffnn_types.h"
#include "../../src/ud_functions/ud_function.h"

namespace bbts {

struct ffnn_add : public ud_impl_t {

  enum class elementwise_fn_type : int32_t {
    NOOP = -1,
    SIGMOID = 0,
    RELU = 1
  };

  // initializes the function
  ffnn_add();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void add(const bbts::ud_impl_t::tensor_params_t &params,
                  const tensor_args_t &_in, tensor_args_t &_out);

};

}