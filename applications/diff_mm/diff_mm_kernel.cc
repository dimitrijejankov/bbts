#include "diff_mm_kernel.h"
#include "../../main/tensor/builtin_formats.h"

template<int kernel_rows, int kernel_cols>
void wierd_kernel(float *a, float *b, float *c, int lda, int ldb, int ldc, int K) {

    // we accumulate stuff here
    __m256 sums[4][kernel_cols / 8] = {};

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < kernel_cols / 8; j++) {

            __m256 b4 = _mm256_load_ps(b + ldb * k + 8 * j);
            for (int i = 0; i < kernel_rows; i++) {
                
                __m256 a4 = _mm256_broadcast_ss(a + i * lda + k);
                auto v = _mm256_sub_ps(a4, b4);
                v = _mm256_mul_ps(v, v);
                sums[i][j] = _mm256_add_ps(v, sums[i][j]);
            }
        }
    }

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0; j < kernel_cols / 8; j++) {
            _mm256_store_ps(&c[i * ldc + j * 8], sums[i][j]);
        }
    }
}

// a(I, K) b(K, J) c(I, J)
template<int kernel_rows, int kernel_cols>
void wierd_mult_with_kernel(float *a, float *b, float *c, int I, int J, int K) {

    assert(I % kernel_rows == 0);
    assert(J % kernel_cols == 0);

    // avx registers are 8 floats wide
    assert(kernel_cols % 8 == 0);

    for (int i = 0; i < I; i += kernel_rows) {

        // #pragma omp parallel for
        for (int j = 0; j < J; j += kernel_cols) {
            wierd_kernel<kernel_rows, kernel_cols>(&a[i * K], &b[j], &c[i * J + j], K, J, J, K);
        }
    }
}

bbts::dense_diff_mm_kernel_t::dense_diff_mm_kernel_t() {

  // set the names
  impl_name = "dense_diff_mm_kernel";
  ud_name = "diff_mm_kernel";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense mult
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_diff_mm_kernel_t::mult;
}

size_t bbts::dense_diff_mm_kernel_t::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) const {
  return 0;
}

size_t bbts::dense_diff_mm_kernel_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                      const bbts::ud_impl_t::meta_args_t &_in) {
  // O(n * m * k)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();
  return 1.45838e-11 * m_a.num_rows * m_a.num_cols * m_b.num_cols;
}

void bbts::dense_diff_mm_kernel_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                             const bbts::ud_impl_t::meta_args_t &_in,
                                             bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  const auto &m_b = _in.get<1>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the output
  m_out = {m_a.num_rows, m_b.num_cols};
}

void bbts::dense_diff_mm_kernel_t::mult(const bbts::ud_impl_t::tensor_params_t &params,
                                     const bbts::ud_impl_t::tensor_args_t &_in,
                                     bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &b = _in.get<1>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // get the sizes
  uint32_t I = m_a.num_rows;
  uint32_t J = m_b.num_cols;
  uint32_t K = m_a.num_cols;

  // make sure the matrix size matches, this is only present during the debug build
  assert(m_a.num_cols == m_b.num_rows);
  
  // run the wierd kernel
  wierd_mult_with_kernel<4, 16>(a.data(), b.data(), out.data(), I, J, K);

  // set the new meta data
  m_out = {m_a.num_rows, m_b.num_cols};
}


