#include "ffnn_diff_mm_kernel.h"
#include "ffnn_mult.h"
#include "ffnn_types.h"

template<int kernel_rows, int kernel_cols>
void wierd_kernel(float *a, float *b, float *c, int lda, int ldb, int ldc, int K) {

    __m256 sums[kernel_rows][kernel_cols] = {};

    for (int k = 0; k < K; k += 8) {
        for (int j = 0; j < kernel_cols; j++) {
            __m256 b4 = _mm256_load_ps(b + ldb * j + k);
            for (int i = 0; i < kernel_rows; i++) {
                __m256 a4 = _mm256_load_ps(a + i * lda + k);
                auto v = _mm256_sub_ps(a4, b4);
                v = _mm256_mul_ps(v, v);
                sums[i][j] = _mm256_add_ps(v, sums[i][j]);
            }
        }
    }

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols; j++) {
            c[i * ldc + j] = sums[i][j][0] + sums[i][j][1] + sums[i][j][2] + sums[i][j][3] +
                             sums[i][j][4] + sums[i][j][5] + sums[i][j][6] + sums[i][j][7];
        }
    }
}


// a(I, K) b(K, J) c(I, J)
template<int kernel_rows, int kernel_cols>
void wierd_mult_with_kernel(float *a, float *b, float *c, int I, int J, int K) {

    assert(I % kernel_rows == 0);
    assert(J % kernel_cols == 0);

    for (int i = 0; i < I; i += kernel_rows) {

        // #pragma omp parallel for
        for (int j = 0; j < J; j += kernel_cols) {
            wierd_kernel<kernel_rows, kernel_cols>(&a[i * K], &b[j * K], &c[i * J + j], K, K, J, K);
        }
    }
}

bbts::ffnn_diff_mm_kernel_t::ffnn_diff_mm_kernel_t() {

  // set the names
  impl_name = "ffnn_diff_mm_kernel";
  ud_name = "ffnn_diff_mm_kernel";

  // set the input and output types
  inputTypes = {"ffnn_dense", "ffnn_dense"};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = false;

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
  wierd_mult_with_kernel<16, 2>(a.data(), b.data(), out.data(), I, I, K);

  // set the new meta data
  m_out = {I, I};
}


