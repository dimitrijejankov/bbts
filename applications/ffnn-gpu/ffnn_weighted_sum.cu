#include "ffnn_types.h"
#include "ffnn_weighted_sum.h"
#include <cassert>

bbts::ffnn_weighted_sum::ffnn_weighted_sum() {

  // set the names
  impl_name = "ffnn_weighted_sum_cpu";
  ud_name = "ffnn_weighted_sum";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU dense add
  is_gpu = true;

  // set the function that actually performs the add
  fn = &ffnn_weighted_sum::add;
}

size_t bbts::ffnn_weighted_sum::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  return 5.91992e-10 * m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_weighted_sum::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the new values
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = m_a.has_bias, 
           .num_aggregated = 1};
}

// kernel definition
__global__ void ffnn_ws_kernel(float *a, float *b, float *c, int n, float lr) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] + lr * b[id];
}

void bbts::ffnn_weighted_sum::add(const bbts::ud_impl_t::tensor_params_t &params,
                                  const bbts::ud_impl_t::tensor_args_t &_in,
                                  bbts::ud_impl_t::tensor_args_t &_out) {

  // get the constants
  auto lr = params.get_float_or_default<0>(1.0f);

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.has_bias == m_b.has_bias);

  // number of thread blocks in grid
  uint32_t block_size = 1024;
  uint32_t n = m_a.num_rows * m_a.num_cols + (m_a.has_bias ? m_a.num_cols : 0);
  uint32_t grid_size = (int)ceil((float)n / block_size);

  // update the weights
  ffnn_ws_kernel<<<grid_size, block_size, 0, params.stream>>>(
    a.data(), b.data(), out.data(), n, lr);

  // set the new meta data
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = m_a.has_bias, 
           .num_aggregated = 1};
}
