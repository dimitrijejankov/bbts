#include <cstdint>
#include <iostream>
#include <string>
#include <dlfcn.h>
#include <vector>
#include "../../src/tensor/tensor.h"
#include "../../src/gpu/scheduler.h"
#include "../../src/tensor/builtin_formats.h"
#include "../../src/utils/terminal_color.h"
#include "../ffnn-gpu/ffnn_add.h"

float learning_rate = 1.0f;

int32_t num_batch = 1280;
int32_t batch_block = 128;

int32_t num_features = 1000;
int32_t features_block = 100;

int32_t num_labels = 1500;
int32_t labels_block = 150;

int32_t embedding_size = 2300;
int32_t embedding_block = 230;

int32_t num_gpus = 4;

bbts::command_id_t cmd_id = 0;
bbts::tid_t currentTID = 0;

using matrix_index = std::map<std::tuple<int32_t, int32_t>, bbts::tid_t>;
using matrix_reduce_index = std::map<std::tuple<int32_t, int32_t>, std::vector<bbts::tid_t>>;

void load_library(const bbts::tensor_factory_ptr_t &tf,
                  const bbts::udf_manager_ptr udf_manager) {

  // open the newly created temporary file
  void* so_handle = dlopen("../libraries/libffnn_gpu_lib.so", RTLD_LOCAL | RTLD_NOW);
  if(!so_handle) {
    std::cout << bbts::red << "Could not open temporary shared library object " << dlerror() << "!\n" << bbts::reset;
    return;
  }

  void* register_tensors_ = dlsym(so_handle, "register_tensors");
  if(register_tensors_) {
    typedef void *register_tensors_f(bbts::tensor_factory_ptr_t);
    auto *register_tensors = (register_tensors_f *) register_tensors_;
    register_tensors(tf);
  }

  // check for the register_udfs
  void* register_udfs_ = dlsym(so_handle, "register_udfs");
  if(register_udfs_) {
    typedef void *register_udfs_f(bbts::udf_manager_ptr);
    auto *register_udfs = (register_udfs_f *) register_udfs_;
    register_udfs(udf_manager);
  }
}

std::vector<std::thread>
run_threads(bbts::multi_gpu_scheduler_ptr_t scheduler,
            bbts::storage_ptr_t storage) {

  std::vector<std::thread> threads;
  threads.push_back(
      std::thread([scheduler]() { scheduler->command_prep_thread(); }));

  threads.push_back(
      std::thread([scheduler]() { scheduler->cpu_to_gpu_thread(); }));

  threads.push_back(std::thread([scheduler, storage]() { 

    while(true) {
      
      // get all the deleted  tensors
      auto deleted_tensors = scheduler->get_deleted_tensors();
      if(deleted_tensors.empty()) {
        break;
      }

      // remove all the tensors
      for(auto t : deleted_tensors) {
        storage->remove_by_tid(t);
      }
    }
  }));

  for (auto dev = 0; dev < scheduler->num_gpus(); ++dev) {

    threads.push_back(
        std::thread([scheduler, dev]() { scheduler->gc_thread(dev); }));

    threads.push_back(std::thread(
        [scheduler, dev]() { scheduler->gpu_execution_thread(dev); }));

    threads.push_back(
        std::thread([scheduler, dev]() { scheduler->gpu_to_gpu_thread(dev); }));
  }

  return std::move(threads);
}

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

matrix_index generate(const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                      const bbts::tensor_factory_ptr_t &factory,
                      const bbts::storage_ptr_t &storage, size_t num_rows, size_t num_cols,
                      size_t split_rows, size_t split_cols) {

  matrix_index out;
  for (size_t rowID = 0; rowID < split_rows; ++rowID) {
    for (size_t colID = 0; colID < split_cols; ++colID) {

      // store the index
      auto tid = currentTID++;
      out[{rowID, colID}] = tid;

      // 
      auto blk_rows = (std::int32_t)(num_rows / split_rows);
      auto blk_cols = (std::int32_t)(num_cols / split_cols);

      // init the tensor
      init_tensor_on_cpu(scheduler, factory, storage, tid,
                         blk_rows, blk_cols, 1.0f);
    }
  }

  return std::move(out);
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

matrix_index generate_multiply(bbts::multi_gpu_scheduler_ptr_t &scheduler,
                               bbts::udf_manager_ptr udf_manager,
                               std::list<bbts::command_ptr_t> &to_schedule,
                               const std::string &ud_name,
                               matrix_index &lhs, matrix_index &rhs,
                               bool lhs_trans, bool rhs_trans, int32_t n,
                               int32_t m, int32_t k,
                               bbts::ffnn_add::elementwise_fn_type final_op) {
  // the parameter data
  std::vector<bbts::command_param_t> param_data = {bbts::command_param_t{.b = lhs_trans},
                                                   bbts::command_param_t{.b = rhs_trans}};
  // make the multiplies
  matrix_reduce_index ridx;
  for (int32_t ni = 0; ni < n; ni++) {
    for (int32_t mi = 0; mi < m; mi++) {
      for (int32_t ki = 0; ki < k; ki++) {

        // get the right row and column id from the left matrix
        auto l_row = lhs_trans ? ki : ni;
        auto l_col = lhs_trans ? ni : ki;

        // get the right row and column id from the right matrix
        auto r_row = rhs_trans ? mi : ki;
        auto r_col = rhs_trans ? ki : mi;

        // store it
        auto tid = currentTID++;
        ridx[{ni, mi}].push_back(tid);

        assert(lhs.find({l_row, l_col}) != lhs.end());
        assert(rhs.find({r_row, r_col}) != rhs.end());

        // make the multiply
        to_schedule.push_back(create_apply(cmd_id, udf_manager, ud_name,  
                                           {lhs[{l_row, l_col}], rhs[{r_row, r_col}]}, 
                                           {tid}, 
                                           {}));
        cmd_id++;
      }
    }
  }

  param_data = {bbts::command_param_t{.i = k},
                bbts::command_param_t{.i = (int32_t)final_op}};

  // make the reduce ops
  matrix_index out;
  for (auto &c : ridx) {

    // set the current tid
    out[c.first] = currentTID;

    // reduce tensors
    to_schedule.push_back(create_reduce(cmd_id,  
                                        udf_manager, 
                                        "ffnn_add", 
                                        c.second, 
                                        currentTID++, 
                                        param_data));
    cmd_id++;

    // delete intermediate
    to_schedule.push_back(create_delete(cmd_id, c.second));
    cmd_id++;
  }

  // return the idex
  return std::move(out);
}

matrix_index apply_binary(bbts::udf_manager_ptr udf_manager, 
                          const std::string &ud_name,
                          std::list<bbts::command_ptr_t> &commands,
                          matrix_index &lhs, matrix_index &rhs,
                          const std::vector<bbts::command_param_t> &param_data) {
  matrix_index out;
  for (auto l : lhs) {

    assert(rhs.find(l.first) != rhs.end());
    auto r = rhs[l.first];

    auto tid = currentTID++;
    out[l.first] = tid;

    commands.push_back(create_apply(cmd_id, udf_manager, ud_name,  
                                    {l.second, r}, 
                                    {tid}, 
                                    param_data));
  }

  return std::move(out);
}

void remove_matrix(const matrix_index &idx,
                   std::list<bbts::command_ptr_t> &commands) {
  
  // store all the ones that need to be deleted
  std::vector<bbts::tid_t> tmp;
  for(auto &it : idx) { tmp.push_back(it.second); }

  // delete intermediate
  commands.push_back(create_delete(cmd_id, tmp));
  cmd_id++;
}

std::vector<bbts::command_ptr_t> to_vector(const std::list<bbts::command_ptr_t> &cmds) {
  std::vector<bbts::command_ptr_t> to_schedule;
  to_schedule.resize(cmds.size());
  auto idx = 0;
  for(auto &cmd : cmds) {
    to_schedule[idx++] = cmd->clone();
  }
  return std::move(to_schedule);
}

int main() {

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;
  config->dev_cluster_ram = 20lu * 1024lu * 1024lu * 1024lu;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udf_manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      num_gpus, 14lu * 1024lu * 1024lu * 1024lu, storage, udf_manager, factory);

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler, storage);

  // load the GPU library
  load_library(factory, udf_manager);

  // generate the input batch
  auto x = generate(scheduler, factory, storage, num_batch,
                    num_features, num_batch / batch_block,
                    num_features / features_block);
  auto y = generate(scheduler, factory, storage, num_batch,
                    num_labels, num_batch / batch_block,
                    num_labels / labels_block);

  // init the weights
  auto w1 = generate(scheduler, factory, storage, 
                     num_features, embedding_size,
                     num_features / features_block, embedding_size / embedding_block);
  auto w2 = generate(scheduler, factory, storage, 
                     embedding_size, num_labels,
                     embedding_size / embedding_block, num_labels / labels_block);

  // a_1 = relu(X * W1 + b)
  std::list<bbts::command_ptr_t> ffnn_commands;
  auto a_1 = generate_multiply(scheduler, udf_manager, ffnn_commands, 
      "ffnn_act_mult", x, w1, false, false,
      num_batch / batch_block, embedding_size / embedding_block,
      num_features / features_block, bbts::ffnn_add::elementwise_fn_type::RELU);

  // a_2 = sigmoid(a_1 * W2 + b)
  auto a_2 = generate_multiply(scheduler, udf_manager, ffnn_commands, 
      "ffnn_act_mult", a_1, w2, false, false,
      num_batch / batch_block, num_labels / labels_block,
      embedding_size / embedding_block, bbts::ffnn_add::elementwise_fn_type::SIGMOID);

  // ∇a_2 = a2 − Y
  std::vector<bbts::command_param_t> param_data = {bbts::command_param_t{.f = -1.0f},
                                                   bbts::command_param_t{.f = 1.0f}};
  auto delta_a_2 =
      apply_binary(udf_manager, "ffnn_weighted_sum", ffnn_commands, y, a_2, param_data);

  // ∇w_2 = a_1^𝑇 * ∇a_2
  auto delta_w_2 = generate_multiply(scheduler, udf_manager, ffnn_commands, 
      "ffnn_back_mult", a_1, delta_a_2, true, false,
      embedding_size / embedding_block, num_labels / labels_block,
      num_batch / batch_block, bbts::ffnn_add::elementwise_fn_type::NOOP);

  // ∇a_2 * W_2^𝑇
  auto delta_a_1_tmp = generate_multiply(scheduler, udf_manager, ffnn_commands, 
      "ffnn_mult", delta_a_2, w2, false, true,
      num_batch / batch_block, embedding_size / embedding_block,
      num_labels / labels_block, bbts::ffnn_add::elementwise_fn_type::NOOP);

  // ∇a_1 = ∇a_2 * W_2^𝑇 .* relu'(a1)
  auto delta_a_1 = apply_binary(udf_manager, "ffnn_matrix_hadamard", ffnn_commands,
                                delta_a_1_tmp, a_1, {});

  // ∇w_1 = x^𝑇 * ∇a_1
  auto delta_w_1 = generate_multiply(scheduler, udf_manager, ffnn_commands, 
      "ffnn_back_mult", x, delta_a_1, true, false,
      num_features / features_block, embedding_size / embedding_block,
      num_batch / batch_block, bbts::ffnn_add::elementwise_fn_type::NOOP);

  // update the weights
  param_data = { bbts::command_param_t{.f = -learning_rate} };
  auto updated_w1 =
      apply_binary(udf_manager, "ffnn_weighted_sum", ffnn_commands, w1, delta_w_1, param_data);
  auto updated_w2 =
      apply_binary(udf_manager, "ffnn_weighted_sum", ffnn_commands, w2, delta_w_2, param_data);

  // do a ton of removes
  remove_matrix(w1, ffnn_commands);
  remove_matrix(w2, ffnn_commands);
  remove_matrix(a_1, ffnn_commands);
  remove_matrix(a_2, ffnn_commands);
  remove_matrix(delta_a_2, ffnn_commands);
  remove_matrix(delta_w_2, ffnn_commands);
  remove_matrix(delta_a_1_tmp, ffnn_commands);
  remove_matrix(delta_a_1, ffnn_commands);
  remove_matrix(delta_w_1, ffnn_commands);

  auto to_schedule = to_vector(ffnn_commands);
  scheduler->schedule(to_schedule);
  
  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  scheduler->save_log("gpu.proto");

  return 0;
}