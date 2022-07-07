
#include "../../src/tensor/tensor_factory.h"
#include "../../src/ud_functions/udf_manager.h"

#include "ffnn_types.h"  

#include "ffnn_activation_mult.h"
#include "ffnn_add.h"
#include "ffnn_matrix_hadamard.h"  
#include "ffnn_mult.h"  
#include "ffnn_uniform_data.h"  
#include "ffnn_uniform_weights.h"  
#include "ffnn_weighted_sum.h"
#include "ffnn_back_mult.h"
#include "ffnn_weighted_sum_sparse_dense.h"
#include "ffnn_uniform_sparse_data.h"

extern "C" {

  void register_tensors(bbts::tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("ffnn_dense", bbts::ffnn_dense_t::get_creation_fs());
    tensor_factory->register_fmt("ffnn_sparse", bbts::ffnn_sparse_t::get_creation_fs());
  }
 
  void register_udfs(bbts::udf_manager_ptr udf_manager) {

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
          bbts::ud_func_t {
            .ud_name = "ffnn_act_mult",
            .is_ass = false,
            .is_com = false,
            .num_in = 2,
            .num_out = 1,
            .impls = {}
          }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_activation_mult>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_add",
        .is_ass = true,
        .is_com = true,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_add>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_matrix_hadamard",
        .is_ass = true,
        .is_com = true,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_matrix_hadamard>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_mult",
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_mult>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_uniform_data",
        .is_ass = false,
        .is_com = false,
        .num_in = 0,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_uniform_data>());


    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_uniform_weights",
        .is_ass = false,
        .is_com = false,
        .num_in = 0,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_uniform_weights>());


    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_weighted_sum",
        .is_ass = true,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_weighted_sum>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_back_mult",
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_back_mult>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_uniform_sparse_data",
        .is_ass = false,
        .is_com = false,
        .num_in = 0,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_uniform_sparse_data>());

    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "ffnn_weighted_sum_sparse_dense",
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
    udf_manager->register_udf_impl(std::make_unique<bbts::ffnn_weighted_sum_sparse_dense>());

  }
}
