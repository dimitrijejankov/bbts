#pragma once

#include "../tensor/tensor_factory.h"
#include "../ud_functions/ud_function.h"
#include <future>
#include <stdexcept>

namespace bbts {

struct null_gpu_scheduler_t {

  // 
  null_gpu_scheduler_t(const bbts::tensor_factory_ptr_t &fact) {};

  // free everything
  ~null_gpu_scheduler_t() {};

  // run the scheduler
  void run() {};

  // runs the kernel
  std::future<bool> execute_kernel(bbts::ud_impl_t* fun,
                                   const bbts::ud_impl_t::tensor_params_t* params,
                                   const bbts::ud_impl_t::tensor_args_t* inputs,
                                   bbts::ud_impl_t::tensor_args_t* outputs) { 
    throw std::runtime_error("Tried to execute a GPU kernel on a node without GPU support!");
  };

  // shutdown the the scheduler
  void shutdown() {};

};

}