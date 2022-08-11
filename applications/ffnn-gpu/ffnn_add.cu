#include "ffnn_add.h"
#include <cmath>
#include <cstdint>

bbts::ffnn_add::ffnn_add() {

  // set the names
  impl_name = "ffnn_add_gpu";
  ud_name = "ffnn_add";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU dense add
  is_gpu = true;

  // set the function that actually performs the add
  fn = &ffnn_add::add;
}

size_t bbts::ffnn_add::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return 5.91992e-10 * m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_add::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  bool has_bias = m_a.has_bias && m_b.has_bias;
  m_out = {.num_rows = m_a.num_rows,
           .num_cols = m_a.num_cols,
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = has_bias,
           .num_aggregated = 1};
}

// kernel definition
__global__ void ffnn_add_kernel(float *a, float *b, float *c, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] + b[id];
}

// kernel definition
__global__ void ffnn_relu_kernel(float *a, float *b, float *c, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n) {
    float x = a[id] + b[id];
    c[id] = x * (x > 0);
  }
}

// kernel definition
__global__ void ffnn_sigmoid_kernel(float *a, float *b, float *c, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n) {
    c[id] = a[id] + b[id];
    c[id] = 1.0f / (1.0f + exp(-c[id]));
  }
}

void bbts::ffnn_add::add(const bbts::ud_impl_t::tensor_params_t &params,
                         const bbts::ud_impl_t::tensor_args_t &_in,
                         bbts::ud_impl_t::tensor_args_t &_out) {

  // get the number of
  auto num_to_reduce = params.get_int_or_default<0>(-1);
  auto elementwise_op = (elementwise_fn_type)params.get_int_or_default<1>(
      (int32_t)elementwise_fn_type::NOOP);

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // set the new meta data
  m_out.num_rows = m_a.num_rows;
  m_out.num_cols = m_a.num_cols;
  m_out.has_bias = m_a.has_bias && m_b.has_bias;
  m_out.num_aggregated = m_a.num_aggregated + m_b.num_aggregated;
  m_out.row_idx = m_a.row_idx;
  m_out.col_idx = m_a.col_idx;

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.has_bias == m_b.has_bias);

  // add a and b
  uint32_t block_size = 1024;

  // number of thread blocks in grid
  uint32_t n = m_a.num_rows * m_a.num_cols;
  uint32_t grid_size = (int)ceil((float)n / block_size);

  // Execute the kernel
  bool run_activation = out.meta().m().num_aggregated == num_to_reduce;
  if (elementwise_op == elementwise_fn_type::RELU && run_activation) {
    ffnn_relu_kernel<<<grid_size, block_size, 0, params.stream>>>(
        a.data(), b.data(), out.data(), n);
  } else if (elementwise_op == elementwise_fn_type::RELU && run_activation) {
    ffnn_sigmoid_kernel<<<grid_size, block_size, 0, params.stream>>>(
        a.data(), b.data(), out.data(), n);
  } else {
    ffnn_add_kernel<<<grid_size, block_size, 0, params.stream>>>(
        a.data(), b.data(), out.data(), n);
  }

  if (m_a.has_bias && m_b.has_bias) {
    n = m_a.num_cols;
    grid_size = (int)ceil((float)m_a.num_cols / block_size);
    ffnn_add_kernel<<<grid_size, block_size, 0, params.stream>>>(
        a.bias(), b.bias(), out.bias(), n);
  }
}