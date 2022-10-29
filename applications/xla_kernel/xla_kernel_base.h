#pragma once

#include <cassert>
#include "../../main/ud_functions/ud_function.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

namespace bbts {

struct RunOptionsHolder {
  RunOptionsHolder(const int num_threads);
  RunOptionsHolder();
  ~RunOptionsHolder();
  
  xla::ExecutableRunOptions* get_run_options();

 private:
  std::unique_ptr<tsl::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
  xla::ExecutableRunOptions run_options;
};

struct xla_kernel_base : public ud_impl_t {

  // initializes the function
  xla_kernel_base(const std::string &hlo);

  // get the required temporary memory
  size_t get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                           const meta_args_t &_in) const override;

  // returns an estimate of the complexity
  size_t get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                             const meta_args_t &_in) override;

  // return the meta of the output
  void get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                    const meta_args_t &_in,
                    meta_args_t &_out) const override;

  // does the work
  void run_me(const bbts::ud_impl_t::tensor_params_t &params,
                     const tensor_args_t &_in,
                     tensor_args_t &_out);

  std::unique_ptr<xla::Executable> executable;

  size_t extra_memory;
  
  std::vector<size_t> input_sizes;

  std::vector<size_t> output_sizes;

};

}