#pragma once

#include <cstddef>
#ifdef ENABLE_GPU
#include <future>
#include <condition_variable>
#include <mutex>
#include <queue>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>
#include <cublasLt.h>
#include "../tensor/builtin_formats.h"
#include "../ud_functions/builtin_functions.h"
#include "../tensor/tensor_factory.h"
#include "../../third_party/cuda/gpu.h"

namespace bbts {

struct gpu_scheduler_impl_t {

  // 
  gpu_scheduler_impl_t(const bbts::tensor_factory_ptr_t &fact);

  // free everything
  ~gpu_scheduler_impl_t();

  // run the scheduler
  void run();

  // runs the kernel
  std::future<bool> execute_kernel(bbts::ud_impl_t* fun,
                                   const bbts::ud_impl_t::tensor_params_t* params,
                                   const bbts::ud_impl_t::tensor_args_t* inputs,
                                   bbts::ud_impl_t::tensor_args_t* outputs);

  // shutdown the the scheduler
  void shutdown();

private:

  // specifies the kernel function to run and the parameters
  struct kernel_spec_t {

    // the kernel we want to run
    bbts::ud_impl_t* fun;

    // the input parameters
    bbts::ud_impl_t::tensor_params_t params;

    // the inputs
    const bbts::ud_impl_t::tensor_args_t* inputs;

    // the outputs
    bbts::ud_impl_t::tensor_args_t* outputs;

    // did succeed running it?
    std::promise<bool> success;
  };
  
  struct gpu_t {
    
    // the kernels we are running  
    kernel_spec_t _specs[3] = {0};

    // the streams and handles FRONT, MID, BACK
    cudaStream_t _streams[3];
    cudaEvent_t  _events[3];
    cublasHandle_t _handles[3];

    // do we have something to do on the FRONT, MID, BACK stream
    bool _has_something[3] = {false, false, false};

    // the stream
    size_t FRONT = 0;
    size_t MID = 1;
    size_t BACK = 2;
  };

  // run the thread for a particular device
  void _run(int device);

  // rotate everything
  void _rotate(size_t &FRONT, size_t &MID, size_t &BACK);

  // used to sync 
  std::mutex _m;
  std::condition_variable _cv;

  // the number of available gpu devices
  int num_devices;

  // the number of devices
  std::vector<gpu_t> devices;

  // the queue of things to process
  std::queue<kernel_spec_t> _q;

  // do we have something to do
  std::uint32_t _left_to_process = 0;

  // is it shutdown
  bool _shutdown = false;

  // the tensor factory
  bbts::tensor_factory_ptr_t _factory;

};

}

#endif