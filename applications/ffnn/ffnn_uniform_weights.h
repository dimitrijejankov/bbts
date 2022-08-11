#pragma once

#include <cassert>
#include "../../src/ud_functions/ud_function.h"

namespace bbts {

struct ffnn_uniform_weights : public ud_impl_t {

  // initializes the function
  ffnn_uniform_weights();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in,
                    meta_args_t &_out) const override;

  // does the work
  static void uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                           const tensor_args_t &_in,
                           tensor_args_t &_out);

};

}