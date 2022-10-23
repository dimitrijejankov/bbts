#pragma once

#include <cassert>
#include "../ud_function.h"

namespace bbts {

struct dense_matrix_mult_t : public ud_impl_t {

  // initializes the function
  dense_matrix_mult_t();

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void mult(const bbts::ud_impl_t::tensor_params_t &params,
                   const tensor_args_t &_in,
                   tensor_args_t &_out);

};

}