#include "ffnn_uniform_weights.h"
#include "ffnn_types.h"
#include <mkl_cblas.h>
#include <mkl.h>

bbts::ffnn_uniform_weights::ffnn_uniform_weights() {

  // set the names
  impl_name = "ffnn_uniform_weights_cpu";
  ud_name = "ffnn_uniform_weights";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = false;

  // set the function that actually performs the add
  fn = &ffnn_uniform_weights::uniform_rand;
}

size_t bbts::ffnn_uniform_weights::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::ffnn_uniform_weights::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::meta_args_t &_in,
                                         bbts::ud_impl_t::meta_args_t &_out) const {

  // get the output argeters
  auto &m_out = _out.get<0>().as<ffnn_dense_meta_t>().m();

  // the number of rows and columns
  auto numRows = (uint32_t) params.get_int<0>();
  auto numCols = (uint32_t) params.get_int<1>();

  // get the row and column indices
  auto rowID = (uint32_t) params.get_int<2>();
  auto colID = (uint32_t) params.get_int<3>();

  // set the new values
  m_out = { .num_rows = numRows,  
            .num_cols = numCols, 
            .row_idx = rowID, 
            .col_idx = colID, 
            .has_bias = true, 
            .num_aggregated = 1 };
}

void bbts::ffnn_uniform_weights::uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {


  // make the random stream
  VSLStreamStatePtr stream;
  auto ret = vslNewStream(&stream, VSL_BRNG_MCG31, time(nullptr));
  assert(VSL_ERROR_OK == ret);

  // get the dense tensor
  auto &out = _out.get<0>().as<ffnn_dense_t>();
  auto &m_out = out.meta().m();

  // set the new meta data
  auto numRows = (uint32_t) params.get_int<0>();
  auto numCols = (uint32_t) params.get_int<1>();

  // get the row and column indices
  auto rowID = (uint32_t) params.get_int<2>();
  auto colID = (uint32_t) params.get_int<3>();

  // the left and right boundary
  auto left = params.get_float_or_default<4>(0.0f);
  auto right = params.get_float_or_default<5>(1.0f);

  // set the new values
  m_out = { .num_rows = numRows,  
            .num_cols = numCols, 
            .row_idx = rowID, 
            .col_idx = colID, 
            .has_bias = true, 
            .num_aggregated = 1 };

  // create a bunch of random numbers
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t) ((numRows  + 1) * numCols), out.data(), left, right);

  // delete the stream
  vslDeleteStream(&stream);
}