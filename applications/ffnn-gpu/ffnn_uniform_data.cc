#include "ffnn_uniform_data.h"
#include "ffnn_types.h"
#include <cstdint>
#include <curand.h>

bbts::ffnn_uniform_data::ffnn_uniform_data() {

  // set the names
  impl_name = "ffnn_uniform_data_cpu";
  ud_name = "ffnn_uniform_data";

  // set the input and output types
  inputTypes = {};
  outputTypes = {"ffnn_dense"};

  // both inputs zero and one can be used as the inplace output
  inputInplace = {};

  // this is a CPU dense add
  is_gpu = true;

  // set the function that actually performs the add
  fn = &ffnn_uniform_data::uniform_rand;
}

size_t bbts::ffnn_uniform_data::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                  const bbts::ud_impl_t::meta_args_t &_in) {

  // make sure that there are enough parameters
  if(params.num_parameters() < 2){
    throw std::runtime_error("Not enough parameters");
  }

  // O(n * m)
  return params.get_uint<0>() * params.get_uint<1>();
}

void bbts::ffnn_uniform_data::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
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
  m_out = {.num_rows = numRows, 
           .num_cols = numCols, 
           .row_idx = rowID,
           .col_idx = colID,
           .has_bias = false, 
           .num_aggregated = 1};
}

void bbts::ffnn_uniform_data::uniform_rand(const bbts::ud_impl_t::tensor_params_t &params,
                                         const bbts::ud_impl_t::tensor_args_t &_in,
                                         bbts::ud_impl_t::tensor_args_t &_out) {


  // make the random stream
  curandGenerator_t gen;
  auto success = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  if(success != CURAND_STATUS_SUCCESS) { throw std::runtime_error("failed to create a curand generator"); }

  success = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  if(success != CURAND_STATUS_SUCCESS) { throw std::runtime_error("failed to set a curand seed"); }

  // get the dense tensor
  auto &out = _out.get<0>().as<ffnn_dense_t>();
  auto &m_out = out.meta().m();

  // the number of rows and columns
  auto numRows = (uint32_t) params.get_int<0>();
  auto numCols = (uint32_t) params.get_int<1>();

  // get the row and column indices
  auto rowID = (uint32_t) params.get_int<2>();
  auto colID = (uint32_t) params.get_int<3>();

  // set the new meta data
  m_out = {.num_rows = numRows, 
           .num_cols = numCols, 
           .row_idx = rowID,
           .col_idx = colID,
           .has_bias = false, 
           .num_aggregated = 1};

  // create a bunch of random numbers
  success = curandGenerateUniform(gen, out.data(), (int32_t) (numRows * numCols));
  if(success != CURAND_STATUS_SUCCESS) { throw std::runtime_error("failed to sample flaots with curand"); }
}