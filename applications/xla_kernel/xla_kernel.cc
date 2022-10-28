#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/compiler/xla/types.h"

#include "xla_kernel.h"

#include "tensorflow/tsl/platform/cpu_info.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"

namespace tos {

RunOptionsHolder::RunOptionsHolder(const int num_threads)
  : pool(new ::tsl::thread::ThreadPool(tsl::Env::Default(), "XLAEigen", num_threads)),
    device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(), pool->NumThreads()))
{
  run_options.set_intra_op_thread_pool(device.get());
}

RunOptionsHolder::RunOptionsHolder()
  : RunOptionsHolder(tsl::port::MaxParallelism())
{}

RunOptionsHolder::~RunOptionsHolder() {}

xla::ExecutableRunOptions* RunOptionsHolder::get_run_options() { 
  return &run_options; 
}

tensorflow::se::DeviceMemoryBase Data::as_tf_data() const {
  return tensorflow::se::DeviceMemoryBase(data, sizeof(float)*size);
}

void Data::_fill(float val) {
  std::fill(data, data + size, val);
}

void Data::_iota() {
  std::iota(data, data + size, 0.0);
}

void Data::_print() {
  for(int i = 0; i != size; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

CpuKernelBla::CpuKernelBla(std::string const& hlo)
{
  std::unique_ptr<xla::HloModule> module = xla::LoadModuleFromData(hlo, "txt").value();
  xla::cpu::CpuCompiler compiler;
  xla::Compiler::CompileOptions dummy;
  executable = compiler.RunBackend(std::move(module), nullptr, dummy).value();

  auto& cpu_executable = this->cpu_executable_();
  scratch_buffer_size_ = 0;
  auto const& allocations = cpu_executable.buffer_assignment().Allocations();
  for(auto const& allocation: allocations) {
    if(!allocation.IsInputOrOutput() &&
       !allocation.is_constant()     &&
       !allocation.is_thread_local())
    {
      scratch_buffer_size_ += allocation.size();
    }
  }
}

CpuKernelBla::~CpuKernelBla() {}

void CpuKernelBla::operator()(
  xla::ExecutableRunOptions* run_options,
  std::vector<Data> const& inns,
  std::vector<Data> const& outs,
  void* scratch_buffer) const
{
  // TODO: figure out how to support multiple outputs
  assert(outs.size() == 1);

  auto& cpu_executable = this->cpu_executable_();

  bool did_allocate = false;

  char* scratch_buffer_char = (char*)scratch_buffer;
  if(scratch_buffer_size_ > 0 && scratch_buffer == nullptr) {
    did_allocate = true;
    scratch_buffer_char = new char[scratch_buffer_size_];
  }

  auto const& allocations = cpu_executable.buffer_assignment().Allocations();

  std::vector<xla::MaybeOwningDeviceMemory> buffers;
  buffers.reserve(allocations.size());

  auto iter_out = outs.begin();
  size_t offset = 0;

  for(auto const& allocation: allocations) {
    if(allocation.IsInputOrOutput()) {
      if(allocation.is_entry_computation_parameter()) {
        auto const& data = inns[allocation.parameter_number()];
	assert(data.size * sizeof(float) >= allocation.size());
        buffers.emplace_back(data.as_tf_data());
      } else {
        auto const& data = *iter_out;
	assert(data.size * sizeof(float) >= allocation.size());
        buffers.emplace_back(data.as_tf_data());
        iter_out++;
      }
    } else if(allocation.is_constant() || allocation.is_thread_local()) {
      buffers.emplace_back(tensorflow::se::DeviceMemoryBase{});
    } else {
      auto sz = allocation.size();
      buffers.emplace_back(tensorflow::se::DeviceMemoryBase{scratch_buffer_char + offset, sz});
      offset += sz;
    }
  }

  auto status = cpu_executable.ExecuteComputeFunction(run_options, buffers, nullptr);

  if(did_allocate) {
    delete[] scratch_buffer_char;
  }
}

void CpuKernelBla::test(bool print) {
  std::vector<Data> inns;
  std::vector<Data> outs;

  auto& cpu_executable = this->cpu_executable_();
  auto const& allocations = cpu_executable.buffer_assignment().Allocations();
  for(auto const& allocation: allocations) {
    // There is only a test to determine if the allocation is an input or an output.
    // Assuming that this is not and in place kernel, then inputs are the read only ones,
    // other wise it is an output.
    if(allocation.IsInputOrOutput()) {
      if(allocation.is_readonly()) {
        auto sz = allocation.size() / sizeof(float);
        inns.push_back(Data{new float[sz], sz});
        inns.back()._fill(1.0);
      } else {
        auto sz = allocation.size() / sizeof(float);
        outs.push_back(Data{new float[sz], sz});
        outs.back()._iota();
      }
    }
  }

  std::cout << "Before test:" << std::endl;
  if(print) {
    for(auto inn: inns) {
      inn._print();
    }
    for(auto out: outs) {
      out._print();
    }
  }

  {
    RunOptionsHolder holder;
    this->operator()(holder.get_run_options(), inns, outs);
  }

  std::cout << "After test:" << std::endl;
  if(print) {
    for(auto inn: inns) {
      inn._print();
    }
    for(auto out: outs) {
      out._print();
    }
  }

  for(auto inn: inns) {
    delete[] inn.data;
  }
  for(auto out: outs) {
    delete[] out.data;
  }
}

} // namespace tos