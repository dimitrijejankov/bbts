#include "xla_kernel_base.h"
#include "xla_types.h"
#include <cstdint>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/cpu_info.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"

bbts::RunOptionsHolder::RunOptionsHolder(const int num_threads)
  : pool(new ::tsl::thread::ThreadPool(tsl::Env::Default(), "XLAEigen", num_threads)),
    device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(), pool->NumThreads()))
{
  run_options.set_intra_op_thread_pool(device.get());
}

bbts::RunOptionsHolder::RunOptionsHolder()
  : RunOptionsHolder(tsl::port::MaxParallelism())
{}

bbts::RunOptionsHolder::~RunOptionsHolder() {}

xla::ExecutableRunOptions* bbts::RunOptionsHolder::get_run_options() { 
  return &run_options; 
}

bbts::xla_kernel_base::xla_kernel_base(const std::string &hlo) {

  // we will fill in these later
  inputTypes = {};
  outputTypes = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually runs the kernel
  fn = [&](const bbts::ud_impl_t::tensor_params_t &params,
          const tensor_args_t &_in,
          tensor_args_t &_out) {
    this->run_me(params, _in, _out);
  };

  // compile the hold executable
  std::unique_ptr<xla::HloModule> mod = xla::LoadModuleFromData(hlo, "txt").value();
  xla::cpu::CpuCompiler compiler;
  xla::Compiler::CompileOptions dummy;
  executable = compiler.RunBackend(std::move(mod), nullptr, dummy).value();

  // go through all the allocations and figure out stuff...
  auto& cpu_executable = *(static_cast<xla::cpu::CpuExecutable*>(executable.get()));
  extra_memory = 0;

  // get all the allocations
  auto const& allocations = cpu_executable.buffer_assignment().Allocations();
  for(auto const& allocation: allocations) {

    // check if this is one of the inputs or outputs 
    if(allocation.IsInputOrOutput()) {

      // check if this is an input
      if(allocation.is_entry_computation_parameter()) {

        auto sz = allocation.size();
        input_sizes.push_back(sz);
        inputTypes.push_back("xla_dense");

      } else {

        // ok it is an output
        // TODO this still might not be an output parameter... need to figure out why....
        auto sz = allocation.size();
        output_sizes.push_back(sz);
        outputTypes.push_back("xla_dense");
      }

    } else if(!allocation.is_constant() && !allocation.is_thread_local()) {

      // constants do not need a buffer as they are resovled by reading the
      // data section of the compiled kernel

      // whether this allocation is used in a parallel calling context such as
      // inside of a map or reduce computation. such allocations need to be thread local. 
      // TODO: figure out why this does not need a buffer
      extra_memory += allocation.size();
    }
  }
}

size_t bbts::xla_kernel_base::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) const {
  return extra_memory;
}

size_t bbts::xla_kernel_base::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {
  // TODO not sure how to figure this out
  return 0;
}

void bbts::xla_kernel_base::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // set the output sizes
  for(auto idx = 0; idx < outputTypes.size(); ++idx) {
    auto &m_out = _out.get_by_idx(idx).as<xla_meta_t>().m();
    m_out.size = output_sizes[idx];
  }
}

void bbts::xla_kernel_base::run_me(const bbts::ud_impl_t::tensor_params_t &params,
                                   const bbts::ud_impl_t::tensor_args_t &_ins,
                                   bbts::ud_impl_t::tensor_args_t &_outs) {

  // TODO: figure out how to support multiple outputs
  assert(_outs.num_args() == 1);

  // this is where all the additional allocated by the kernel will reside
  char* tmp_buffer = static_cast<char*>(params.get_additional_memory());

  // grab compile CPU kernel
  auto& cpu_executable = *(static_cast<xla::cpu::CpuExecutable*>(executable.get()));
  auto const& allocations = cpu_executable.buffer_assignment().Allocations();

  // these are all the buffers the kernel will use
  std::vector<xla::MaybeOwningDeviceMemory> buffers;
  buffers.reserve(allocations.size());

  // all the offsets
  auto in_idx = 0;
  auto out_idx = 0;
  size_t tmp_buffer_offset = 0;

  // go through all the allocations and reserve and set the appropriate buffer
  for(auto const& allocation: allocations) {

    // check if this is an input or output
    if(allocation.IsInputOrOutput()) {
      
      // check if this is an input
      if(allocation.is_entry_computation_parameter()) {
        
        // set the tensor as buffer
        auto &in = _ins.get_by_idx(in_idx++).as<xla_tensor_t>();
        buffers.emplace_back(in.as_tf_data());

      } else {

        // TODO this needs more investigation
        // set the tensor as buffer
        auto &out = _outs.get_by_idx(out_idx++).as<xla_tensor_t>();
        buffers.emplace_back(out.as_tf_data());
      }

    } else if(allocation.is_constant() || allocation.is_thread_local()) {
      buffers.emplace_back(tensorflow::se::DeviceMemoryBase{});
    } else {

      auto sz = allocation.size();
      buffers.emplace_back(tensorflow::se::DeviceMemoryBase{tmp_buffer + tmp_buffer_offset, sz});
      tmp_buffer_offset += sz;
    }
  }

  // finaly run the kernel
  RunOptionsHolder run_options;
  auto status = cpu_executable.ExecuteComputeFunction(run_options.get_run_options(), buffers, nullptr);
}