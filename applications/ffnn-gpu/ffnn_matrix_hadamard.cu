#include "ffnn_matrix_hadamard.h"
#include "ffnn_types.h"

bbts::ffnn_matrix_hadamard::ffnn_matrix_hadamard() {

  // set the names
  impl_name = "ffnn_matrix_hadamard_cpu";
  ud_name = "ffnn_matrix_hadamard";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU kernel
  is_gpu = true;

  // set the function that actually performs the product
  fn = &ffnn_matrix_hadamard::mult;
}

size_t bbts::ffnn_matrix_hadamard::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();

  // this is not exactly correct but I use it because I am lazy...
  return 5.91992e-10 * m_a.num_rows * m_a.num_cols;
}

void bbts::ffnn_matrix_hadamard::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
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
           .has_bias = true, 
           .num_aggregated = 1};
}

__global__ void ffnn_relu_dif_kernel(float *a, float *b, float *out, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n) {
    float x = a[id] * b[id];
    out[id] = x > 0.0f;
  }
}

__global__ void ffnn_elementwise_mult_kernel(float *a, float *b, float *c, int n) {

  // get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] * b[id];
}

void bbts::ffnn_matrix_hadamard::mult(const bbts::ud_impl_t::tensor_params_t &params,
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

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.has_bias == m_b.has_bias);

  // set the new meta data
  m_out = {.num_rows = m_a.num_rows, 
           .num_cols = m_a.num_cols, 
           .row_idx = m_a.row_idx,
           .col_idx = m_a.col_idx,
           .has_bias = true, 
           .num_aggregated = 1};

  // multiply a and b
  auto out_data = out.data();
  auto a_data = a.data();
  auto b_data = b.data();

  // add a and b
  uint32_t block_size = 1024;

  // number of thread blocks in grid
  uint32_t n = m_a.num_rows * m_a.num_cols;
  uint32_t grid_size = (int)ceil((float)n / block_size);

  ffnn_relu_dif_kernel<<<grid_size, block_size, 0, params.stream>>>(
    a_data, b_data, out_data, n);

  // multiply the bias
  if(m_a.has_bias) {
    n = m_a.num_cols;
    grid_size = (int)ceil((float)m_a.num_cols / block_size);
    ffnn_elementwise_mult_kernel<<<grid_size, block_size, 0, params.stream>>>(
        a.bias(), b.bias(), out.bias(), n);
  }
}
