#pragma once

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

#include <iostream>
#include <memory>
#include <vector>

#include <algorithm> // fill
#include <numeric>   // iota

namespace Eigen {
struct ThreadPoolDevice;
}

namespace tos {

struct Data {
  float* data;
  size_t size;

  tensorflow::se::DeviceMemoryBase as_tf_data() const;

  void _fill(float val);
  void _iota();
  void _print();
};

struct CpuKernelBla {
  CpuKernelBla(std::string const& hlo);
  
  ~CpuKernelBla();

  void operator()(
    xla::ExecutableRunOptions* run_options,
    std::vector<Data> const& inns,
    std::vector<Data> const& outs,
    void* scratch_buffer = nullptr) const;

  void test(bool print=true);

  size_t scratch_buffer_size() const {
    return scratch_buffer_size_;
  }

private:
  std::unique_ptr<xla::Executable> executable;
  size_t scratch_buffer_size_;

  xla::cpu::CpuExecutable& cpu_executable_() const {
    return *(static_cast<xla::cpu::CpuExecutable*>(executable.get()));
  }
};

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

} // namespace tos