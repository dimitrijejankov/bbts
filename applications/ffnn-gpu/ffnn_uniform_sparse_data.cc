#include "ffnn_uniform_sparse_data.h"
#include "ffnn_types.h"
#include <mkl_cblas.h>
#include <mkl.h>
#include <random>

bbts::ffnn_uniform_sparse_data::ffnn_uniform_sparse_data() {

  // set the names
  impl_name = "ffnn_uniform_sparse_data_cpu";
  ud_name = "ffnn_uniform_sparse_data";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"ffnn_sparse"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU sparse op
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_uniform_sparse_data::uniform_rand;
}

size_t bbts::ffnn_uniform_sparse_data::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                           const bbts::ud_impl_t::meta_args_t &_in) {
  return 0;
}

size_t bbts::ffnn_uniform_sparse_data::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::ffnn_uniform_sparse_data::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_sparse_meta_t>().m();

  // set the new values (we add a new non zero value to each row)
  m_out = { .num_rows = params.get_uint<0>(),  .num_cols = params.get_uint<1>(), .nnz = params.get_uint<0>()};
}

void bbts::ffnn_uniform_sparse_data::uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {


  // get the dense tensor
  auto &out = _out.get<0>().as<ffnn_sparse_t>();
  auto &m_out = out.meta().m();

  // the number of rows and columns
  auto numRows = (uint32_t) params.get_int<0>();
  auto numCols = (uint32_t) params.get_int<1>();

  // the left and right boundary
  auto left = params.get_float_or_default<2>(0.0f);
  auto right = params.get_float_or_default<3>(1.0f);

  // set the new meta data
  m_out = { .num_rows = params.get_uint<0>(),  .num_cols = params.get_uint<1>(), .nnz = params.get_uint<0>()};

  std::default_random_engine generator;
  std::uniform_int_distribution<int> col_dist(0, m_out.num_cols);
  std::uniform_real_distribution<float> val_dist(left, right);

  // fill in the data
  auto out_data = out.data();
  for(auto row = 0; row < m_out.num_rows; ++row) {
    out_data[row].val = val_dist(generator);
    out_data[row].row = row;
    out_data[row].col = col_dist(generator);
  }
}