#include "mm_util.h"
#include <utility>

void init_tensor_on_cpu(const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                        const bbts::tensor_factory_ptr_t &factory,
                        const bbts::storage_ptr_t &storage, bbts::tid_t tid,
                        uint32_t num_rows, uint32_t num_cols, float value) {

  // make the meta
  bbts::dense_tensor_meta_t dm{tid, num_rows, num_cols};
  dm.fmt_id = factory->get_tensor_ftm("dense");
  auto &m = dm.as<bbts::tensor_meta_t>();

  // get how much we need to allocate
  auto num_bytes = factory->get_tensor_size(m);

  // make the tensor and const init it
  storage->local_transaction(
      {}, {{tid, num_bytes}},
      [&](const bbts::storage_t::reservation_result_t &res) {

        // create the tensor
        auto &ts = res.create[0].get().tensor->as<bbts::dense_tensor_t>();
        for (auto idx = 0; idx < num_rows * num_cols; ++idx) {
          ts.data()[idx] = value;
        }

        ts.get_meta<bbts::tensor_meta_t>() = m;
      });

  // mark the that the tensor is on the CPU
  scheduler->mark_tensor_on_cpu(tid, num_bytes, m);
}

void init_blocked_matrix(float &val,
                         bbts::tid_t &cur_idx,
                         matrix_index_t &index, 
                         const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                         const bbts::tensor_factory_ptr_t &factory,
                         const bbts::storage_ptr_t &storage,
                         size_t row_blocking, 
                         size_t row_block_size,
                         size_t col_blocking, 
                         size_t col_block_size) {
  
  for(auto row_idx = 0; row_idx < row_blocking; ++row_idx) {
    for(auto col_idx = 0; col_idx < col_blocking; ++col_idx) {
      
      init_tensor_on_cpu(scheduler, factory, storage, cur_idx,
                         row_block_size, col_block_size, val);
      std::get<0>(index[{row_idx, col_idx}]) = cur_idx++;
      std::get<1>(index[{row_idx, col_idx}]) = val;
      val += 0.001f;
    }
  }
}

bbts::command_ptr_t
create_apply(bbts::command_id_t id,
             bbts::udf_manager_ptr udm, const std::string &ud_name,
             const std::vector<bbts::tid_t> &inputs,
             const std::vector<bbts::tid_t> &outputs,
             const std::vector<bbts::command_param_t> &params) {

  std::vector<bbts::command_t::tid_node_id_t> prep_in;
  std::vector<std::string> input_types;
  for (auto in : inputs) {
    prep_in.push_back(bbts::command_t::tid_node_id_t{.tid = in, .node = 0});
    input_types.push_back("dense");
  }

  std::vector<bbts::command_t::tid_node_id_t> prep_out;
  std::vector<std::string> output_types;
  for (auto out : outputs) {
    prep_out.push_back(bbts::command_t::tid_node_id_t{.tid = out, .node = 0});
    output_types.push_back("dense");
  }
  auto matcher = udm->get_matcher_for(ud_name);

  #ifdef ENABLE_GPU
  bool is_gpu = true;
  #else
  bool is_gpu = false;
  #endif

  auto ud = matcher->findMatch({}, output_types, is_gpu);
  auto cmd = bbts::command_t::create_apply(id, ud->impl_id, is_gpu, params,
                                           prep_in, prep_out);
  return std::move(cmd);
}

bbts::command_ptr_t
create_reduce(bbts::command_id_t id,
              bbts::udf_manager_ptr udm, const std::string &ud_name,
              const std::vector<bbts::tid_t> &inputs,
              bbts::tid_t output,
              const std::vector<bbts::command_param_t> &params) {

  std::vector<bbts::command_t::tid_node_id_t> prep_in;
  std::vector<std::string> input_types;
  for (auto in : inputs) {
    prep_in.push_back(bbts::command_t::tid_node_id_t{.tid = in, .node = 0});
    input_types.push_back("dense");
  }

  #ifdef ENABLE_GPU
  bool is_gpu = true;
  #else
  bool is_gpu = false;
  #endif

  std::vector<bbts::command_t::tid_node_id_t> prep_out;
  auto matcher = udm->get_matcher_for(ud_name);
  auto ud = matcher->findMatch(input_types, {"dense"}, is_gpu);
  auto cmd = bbts::command_t::create_reduce(id, ud->impl_id, is_gpu, params,
                                            prep_in, bbts::command_t::tid_node_id_t{.tid = output, .node = 0});
  return std::move(cmd);
}

bbts::command_ptr_t
create_delete(bbts::command_id_t id, const std::vector<bbts::tid_t> &inputs) {

  std::vector<bbts::command_t::tid_node_id_t> prep_in;
  for (auto in : inputs) {
    prep_in.push_back(bbts::command_t::tid_node_id_t{.tid = in, .node = 0});
  }
  auto cmd = bbts::command_t::create_delete(id, prep_in);
  return std::move(cmd);
}

std::vector<bbts::command_ptr_t> make_multiply(bbts::tid_t &cur_tid,
                                               bbts::udf_manager_ptr udf_manager,
                                               matrix_index_t &a_index, 
                                               matrix_index_t &b_index, 
                                               matrix_index_t &c_index, 
                                               size_t matrix_blocking, 
                                               size_t matrix_block_size) {

  bbts::command_id_t cmd_id = 0;
  matrix_reduce_index_t reduce_idx;
  auto total = matrix_blocking * matrix_blocking * matrix_blocking + 
               2 * matrix_blocking * matrix_blocking;
  std::vector<bbts::command_ptr_t> to_schedule(total);

  for(auto i = 0; i < matrix_blocking; ++i) {
    for(auto j = 0; j < matrix_blocking; ++j) {
      for(auto k = 0; k < matrix_blocking; ++k) {

        // make the multiply
        std::get<1>(reduce_idx[{i, j}]).push_back(cur_tid);
        to_schedule[cmd_id] = create_apply(cmd_id, 
                                           udf_manager, 
                                           "matrix_mult",  
                                           {std::get<0>(a_index[{i, k}]), 
                                            std::get<0>(b_index[{k, j}])}, 
                                           {cur_tid++}, 
                                           {});
        std::get<0>(reduce_idx[{i, j}]) += std::get<1>(a_index[{i, k}]) * 
                                           std::get<1>(b_index[{k, j}]) *
                                           matrix_block_size;
        cmd_id++;
      }
    }
  }

  for(auto i = 0; i < matrix_blocking; ++i) {
    for(auto j = 0; j < matrix_blocking; ++j) {

      // reduce tensors
      std::get<0>(c_index[{i, j}]) = cur_tid;
      std::get<1>(c_index[{i, j}]) = std::get<0>(reduce_idx[{i, j}]);
      to_schedule[cmd_id] = create_reduce(cmd_id,  
                                          udf_manager, 
                                          "matrix_add", 
                                          std::get<1>(reduce_idx[{i, j}]), 
                                          cur_tid++, 
                                          {});
      cmd_id++;

      // delete intermediate
      to_schedule[cmd_id] = create_delete(cmd_id, std::get<1>(reduce_idx[{i, j}]));
      cmd_id++;
    }
  }

  return std::move(to_schedule);
}

std::vector<bbts::command_ptr_t> delete_matrix(matrix_index_t &index) {
  std::vector<bbts::tid_t> tids;
  for(auto &idx : index) {
    tids.push_back(std::get<0>(idx.second));
  }
  std::vector<bbts::command_ptr_t> tmp(1);
  tmp[0] = create_delete(0, tids); 
  return std::move(tmp);
}