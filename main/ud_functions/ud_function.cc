#include "ud_function.h"

void bbts::ud_impl_t::call_ud(const bbts::ud_impl_t::tensor_params_t &_params,
                              const tensor_args_t &_in,
                              tensor_args_t &_out) const {

    // just call the function
    fn(_params, _in, _out);
}

void bbts::ud_impl_t::call_gpu_ud(const bbts::ud_impl_t::tensor_params_t &_params,
                                  const tensor_args_t &_in,
                                  tensor_args_t &_out) const {
    // just call the kernel
    gpu_fn(_params, _in, _out);                          
}