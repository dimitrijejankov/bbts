#include "ffnn_diff_mm_kernel.h"
#include "ffnn_mult.h"
#include "ffnn_types.h"

// TODO this is a silly kernel but tensorflow will not do any better so I will go with it...
__global__ void mm_kernel(float const* in_1, float const* in_2, float* out, size_t N, size_t K, size_t M)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t n{blockIdx.y * blockDim.y + threadIdx.y};
    size_t m{blockIdx.x * blockDim.x + threadIdx.x};

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    if ((n >= N) || (m >= M))
    {
        return;
    }

    float acc_sum{0};
    for (size_t k{0}; k < K; ++k)
    {
        float tmp = (in_1[n * K + k] - in_2[m * K + k]);
        acc_sum += tmp * tmp;
    }
    out[n * M + m] = acc_sum;
}

bbts::ffnn_diff_mm_kernel_t::ffnn_diff_mm_kernel_t() {

  // set the names
  impl_name = "ffnn_diff_mm_kernel_gpu";
  ud_name = "ffnn_diff_mm_kernel";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = true;

  // set the function that actually performs the add
  fn = &ffnn_diff_mm_kernel_t::mult;
}

size_t bbts::ffnn_diff_mm_kernel_t::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) const {
  return 0;
}

size_t bbts::ffnn_diff_mm_kernel_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {
  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();
  return 1.45838e-11 * m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::ffnn_diff_mm_kernel_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                             const bbts::ud_impl_t::meta_args_t &_in,
                                             bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<ffnn_dense_meta_t>().m();
  const auto &m_b = _in.get<1>().as<ffnn_dense_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // get the sizes
  uint32_t I = m_a.num_rows;

  // set the output
  m_out = {I, I};
}

void bbts::ffnn_diff_mm_kernel_t::mult(const bbts::ud_impl_t::tensor_params_t &params,
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
  uint32_t K = m_a.num_cols;

  // make sure the matrix size matches, this is only present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  
  // run the wierd kernel
  #define ___BLOCK_DIM 32
  dim3 threads_per_block(___BLOCK_DIM, ___BLOCK_DIM);
  dim3 blocks_per_grid(1, 1);
  blocks_per_grid.x = std::ceil(static_cast<double>(K) / static_cast<double>(threads_per_block.x));
  blocks_per_grid.y = std::ceil(static_cast<double>(K) / static_cast<double>(threads_per_block.y));
  mm_kernel<<<blocks_per_grid, threads_per_block, 0, params.stream>>>(a.data(), b.data(), out.data(), K, I, K);
  #undef ___BLOCK_DIM

  // set the new meta data
  m_out = {I, I};
}


