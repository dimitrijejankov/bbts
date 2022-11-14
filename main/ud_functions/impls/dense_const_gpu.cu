#include "../../tensor/builtin_formats.h"
#include "dense_const_gpu.h"

bbts::dense_const_gpu::dense_const_gpu() {

  // set the names
  impl_name = "dense_const";
  ud_name = "const";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = true;

  // set the function that actually performs the add
  fn = &dense_const_gpu::uniform_rand;
}

size_t bbts::dense_const_gpu::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  auto num_rows = (uint32_t)params.get_int<0>();
  auto num_cols = (uint32_t)params.get_int<1>();
  return num_rows * num_cols;
}

void bbts::dense_const_gpu::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  // the number of rows and columns
  auto num_rows = (uint32_t)params.get_int<0>();
  auto num_cols = (uint32_t)params.get_int<1>();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = {num_rows, num_cols};
}

size_t bbts::dense_const_gpu::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) const {
  return 0;
}

// kernel definition
__global__ void dense_uniform_fun(float *out, int n, float v) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    out[id] = v;
}

void bbts::dense_const_gpu::uniform_rand(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::tensor_args_t &_in,
    bbts::ud_impl_t::tensor_args_t &_out) {

  dense_tensor_t &out = _out.get<0>().as<dense_tensor_t>();

  // get the number of rows and columns
  auto I = (uint32_t)params.get_int<0>();
  auto J = (uint32_t)params.get_int<1>();
  auto value = (uint32_t)params.get_float_or_default<2>(0.0f);

  // number of threads in each thread block
  uint32_t block_size = 1024;

  // number of thread blocks in grid
  uint32_t n = I * J;
  uint32_t grid_size = (int)ceil((float)n / block_size);

  // execute the kernel
  dense_uniform_fun<<<grid_size, block_size, 0, params.stream>>>(out.data(), n, value);

  // set the meta
  out.meta().m() = {I, J};
}