#include "ffnn_activation_mult.h"
#include "ffnn_types.h"
#include <cassert>
#include <cublas_v2.h>

bbts::ffnn_activation_mult::ffnn_activation_mult() {

  // set the names
  impl_name = "ffnn_act_mult_gpu";
  ud_name = "ffnn_act_mult";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = true;

  // set the function that actually performs the add
  fn = &ffnn_activation_mult::mult;
}

size_t bbts::ffnn_activation_mult::get_complexity_hint(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();
  return 1.45838e-11 * m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

__global__ void ffnn_activation_mult_bias_add_kernel(float *b, float *c,
                                                     int num_rows,
                                                     int num_cols) {

  // get our global thread ID
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  // make sure we do not go out of bounds
  if (row < num_rows && col < num_cols)
    c[col * num_rows + row] += b[col];
}

void bbts::ffnn_activation_mult::get_out_meta(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::meta_args_t &_in,
    bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // set the output
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  m_out = {.num_rows = I,
           .num_cols = J,
           .row_idx = m_a.row_idx,
           .col_idx = m_b.col_idx,
           .has_bias = false,
           .num_aggregated = 1};

  auto num_elements = m_out.num_cols * m_out.num_rows;
  num_elements += m_out.has_bias ? m_out.num_cols : 0;
}

void bbts::ffnn_activation_mult::mult(
    const bbts::ud_impl_t::tensor_params_t &params,
    const bbts::ud_impl_t::tensor_args_t &_in,
    bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<ffnn_dense_t>();
  auto &b = _in.get<1>().as<ffnn_dense_t>();
  auto &out = _out.get<0>().as<ffnn_dense_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // get the sizes
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  uint32_t K = m_a.num_cols;

  // make sure the matrix size matches,
  // this is only present during the debug build
  assert(m_a.num_cols == m_b.num_rows);
  assert(m_b.has_bias);

  // get the ptrs
  float *outData = out.data();
  float *in1Data = a.data();
  float *in2Data = b.data();

  // set the new meta data
  m_out = {.num_rows = I,
           .num_cols = J,
           .row_idx = m_a.row_idx,
           .col_idx = m_b.col_idx,
           .has_bias = false,
           .num_aggregated = 1};

  // do the multiply
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(params.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, I, J, K, &alpha,
              in1Data, I, in2Data, K, &beta, outData, I);

  if (m_a.col_idx == 0 && m_b.row_idx == 0) {

    dim3 threadsPerBlock(8, 8);
    dim3 block_size((int)ceil((float)I / 8), (int)ceil((float)J / 8));

    ffnn_activation_mult_bias_add_kernel<<<block_size, threadsPerBlock, 0,
                                           params.stream>>>(b.bias(),
                                                            out.data(), I, J);
  }
}