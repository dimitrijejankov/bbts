#include "dense_matrix_relu.h"
#include "../../tensor/builtin_formats.h"

bbts::dense_matrix_relu_t::dense_matrix_relu_t() {

  // set the names
  impl_name = "dense_matrix_relu";
  ud_name = "matrix_relu";

  // set the input and output types
  inputTypes = {"dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {0};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_matrix_relu_t::relu;
}

size_t bbts::dense_matrix_relu_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                     const bbts::ud_impl_t::meta_args_t &_in) {

  // O(n * m)
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
  return m_a.num_rows * m_a.num_cols;
}

void bbts::dense_matrix_relu_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                            const bbts::ud_impl_t::meta_args_t &_in,
                                            bbts::ud_impl_t::meta_args_t &_out) const {

  // get the input argeters
  const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();

  // get the output argeters
  auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

  // set the new values
  m_out = { m_a.num_rows, m_a.num_cols };
}

void bbts::dense_matrix_relu_t::relu(const bbts::ud_impl_t::tensor_params_t &params,
                                     const bbts::ud_impl_t::tensor_args_t &_in,
                                     bbts::ud_impl_t::tensor_args_t &_out) {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_out = out.meta().m();

  // 
  float *out_data =  out.data();
  float *a_data   =  a.data();

  // 
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out_data[row * m_a.num_cols + col] = a_data[row * m_a.num_cols + col] > 0 ? a_data[row * m_a.num_cols + col] : 0;
    }
  }

  // set the new meta data
  m_out = {m_a.num_rows, m_a.num_cols};
}
