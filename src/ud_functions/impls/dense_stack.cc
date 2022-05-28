#include "dense_stack.h"
#include "../../tensor/builtin_formats.h"
#include <cstdint>
#include <mkl/mkl_cblas.h>
#include <mkl/mkl.h>

bbts::dense_stack_t::dense_stack_t() {

  // set the names
  impl_name = "dense_stack";
  ud_name = "stack";

  // set the input and output types
  inputTypes = {"dense", "dense"};
  outputTypes = {"dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &dense_stack_t::apply;
}

size_t bbts::dense_stack_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
  
}

void bbts::dense_stack_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the tensors as dense tensors
  auto &a = _in.get<0>().as<dense_tensor_t>();
  auto &b = _in.get<1>().as<dense_tensor_t>();
  auto &out = _out.get<0>().as<dense_tensor_t>();

  // get the meta for the tensors
  auto &m_a = a.meta().m();
  auto &m_b = b.meta().m();
  auto &m_out = out.meta().m();

  // make sure the matrix size matches
  assert(m_a.num_cols == m_b.num_cols);

  // set the new meta data
  m_out = {m_a.num_rows + m_b.num_rows, m_a.num_cols};
}

void bbts::dense_stack_t::apply(const bbts::ud_impl_t::tensor_params_t &params,
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

  // make sure the matrix size matches
  assert(m_a.num_cols == m_b.num_cols);

  // stack a and b
  for (auto row = 0; row < m_a.num_rows; ++row) {
    for (auto col = 0; col < m_a.num_cols; ++col) {
      out.data()[row * m_a.num_cols + col] = a.data()[row * m_a.num_cols + col];
    }
  }

  for (auto row = 0; row < m_b.num_rows; ++row) {
    for (auto col = 0; col < m_b.num_cols; ++col) {
      out.data()[m_a.num_rows * m_a.num_cols + row * m_b.num_cols +  col] = b.data()[row * m_b.num_cols + col];
    }
  }

  // set the new meta data
  m_out = {m_a.num_rows + m_b.num_rows, m_a.num_cols};
}