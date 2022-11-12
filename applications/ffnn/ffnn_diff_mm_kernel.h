#pragma once

#include <cassert>
#include <memory>
#include <iomanip>
#include <immintrin.h>

#include "../../main/ud_functions/ud_function.h"

namespace bbts {

struct ffnn_diff_mm_kernel_t : public ud_impl_t {

  // initializes the function
  ffnn_diff_mm_kernel_t();

  // get the required temporary memory
  size_t get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                           const meta_args_t &_in) const override;

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in, meta_args_t &_out) const override;

  // does the work
  static void mult(const bbts::ud_impl_t::tensor_params_t &params,
                  const tensor_args_t &_in, tensor_args_t &_out);

};

}