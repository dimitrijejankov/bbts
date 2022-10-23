#include "dense_matrix_hadamard.h"
#include "../../tensor/builtin_formats.h"

bbts::dense_matrix_hadamard_t::dense_matrix_hadamard_t() {

  // set the names
  impl_name = "dense_matrix_hadamard";
  ud_name = "matrix_matrix_hadamard";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0, 1};

  // this is a CPU kernel
  is_gpu = false;

  // set the function that actually performs the product
  fn = &dense_matrix_hadamard_t::mult;
}

size_t bbts::dense_matrix_hadamard_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::dense_matrix_hadamard_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols };
}

void bbts::dense_matrix_hadamard_t::mult(const bbts::ud_impl_t::tensor_params_t &params,
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

  // make sure the matrix size matches, this is only
  // present during the debug build
  assert(m_a.num_rows == m_b.num_rows);
  assert(m_a.num_cols == m_b.num_cols);
  assert(m_a.num_rows == m_out.num_rows);
  assert(m_a.num_cols == m_out.num_cols);

  // add a and b
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.data()[row * m_a.num_cols + col] = a.data()[row * m_a.num_cols + col] *
                                             b.data()[row * m_a.num_cols + col];
    }
  }

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols};
}
