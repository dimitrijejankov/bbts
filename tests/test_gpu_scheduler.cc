#include "../src/gpu/scheduler.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

using namespace std::chrono;

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
  auto ud = matcher->findMatch({}, output_types, true);
  auto cmd = bbts::command_t::create_apply(id, ud->impl_id, true, params,
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

  std::vector<bbts::command_t::tid_node_id_t> prep_out;
  auto matcher = udm->get_matcher_for(ud_name);
  auto ud = matcher->findMatch(input_types, {"dense"}, true);
  auto cmd = bbts::command_t::create_reduce(id, ud->impl_id, true, params,
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

std::vector<std::thread>
run_threads(bbts::multi_gpu_scheduler_ptr_t scheduler) {

  std::vector<std::thread> threads;
  threads.push_back(
      std::thread([scheduler]() { scheduler->command_prep_thread(); }));

  threads.push_back(
      std::thread([scheduler]() { scheduler->cpu_to_gpu_thread(); }));

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

TEST(TestGPUScheduler, Test1) {

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udf_manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      4, 16lu * 1024lu * 1024lu * 1024lu, storage, udf_manager, factory);

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler);

  // schedule the run commands
  std::vector<bbts::command_ptr_t> to_schedule(1);
  to_schedule[0] = std::move(create_apply(0, udf_manager, "const", {}, {0},
                                          { bbts::command_param_t{.i = 100}, 
                                            bbts::command_param_t{.i = 100}, 
                                            bbts::command_param_t{.f = 2.0f} }));
  scheduler->schedule(to_schedule);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  // check the created tensor
  storage->local_transaction(
      {0}, {}, [&](const bbts::storage_t::reservation_result_t &res) {
        auto ts = res.get[0].get().tensor;
        auto &t = ts->as<bbts::dense_tensor_t>();
        for (auto idx = 0; idx < 100 * 100; ++idx) {
          EXPECT_NEAR(t.data()[idx], 2.0f, 0.001);
        }
      });
}

TEST(TestGPUScheduler, Test2) {

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udf_manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      4, 16lu * 1024lu * 1024lu * 1024lu, storage, udf_manager, factory);

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler);

  // create four tensors on the CPU
  init_tensor_on_cpu(scheduler, factory, storage, 0, 100, 100, 1.0f);
  init_tensor_on_cpu(scheduler, factory, storage, 1, 100, 100, 2.0f);
  init_tensor_on_cpu(scheduler, factory, storage, 2, 100, 100, 3.0f);
  init_tensor_on_cpu(scheduler, factory, storage, 3, 100, 100, 4.0f);

  // create a reduce and schedule it
  std::vector<bbts::command_ptr_t> to_schedule(1);
  auto cmd = create_reduce(0, udf_manager, "matrix_add", {0, 1, 2, 3}, 4, {});
  to_schedule[0] = std::move(cmd);
  scheduler->schedule(to_schedule);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  storage->local_transaction(
      {4}, {}, [&](const bbts::storage_t::reservation_result_t &res) {
        auto ts = res.get[0].get().tensor;
        auto &t = ts->as<bbts::dense_tensor_t>();
        for (auto idx = 0; idx < 100 * 100; ++idx) {
          EXPECT_NEAR(t.data()[idx], 10.0f, 0.001);
        }
      });
}

TEST(TestGPUScheduler, Test3) {

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udf_manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      4, 16lu * 1024lu * 1024lu * 1024lu, storage, udf_manager, factory);

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler);

  // create four tensors on the CPU for matrix A
  init_tensor_on_cpu(scheduler, factory, storage, 0, 100, 100, 1.0f); // A(0, 0)
  init_tensor_on_cpu(scheduler, factory, storage, 1, 100, 100, 2.0f); // A(1, 0)
  init_tensor_on_cpu(scheduler, factory, storage, 2, 100, 100, 3.0f); // A(0, 1)
  init_tensor_on_cpu(scheduler, factory, storage, 3, 100, 100, 4.0f); // A(1, 1)

  // create four tensors on the CPU for matrix B
  init_tensor_on_cpu(scheduler, factory, storage, 4, 100, 100, 2.0f); // B(0, 0)
  init_tensor_on_cpu(scheduler, factory, storage, 5, 100, 100, 3.0f); // B(1, 0)
  init_tensor_on_cpu(scheduler, factory, storage, 6, 100, 100, 4.0f); // B(0, 1)
  init_tensor_on_cpu(scheduler, factory, storage, 7, 100, 100, 5.0f); // B(1, 1)

  // create a reduce and schedule it
  auto cmd1 = create_apply(0, udf_manager, "matrix_mult",  {0, 4}, {8}, {});    // C_0(0, 0) = A(0, 0) * B(0, 0)
  auto cmd2 = create_apply(1, udf_manager, "matrix_mult",  {2, 5}, {9}, {});    // C_1(0, 0) = A(0, 1) * B(1, 0)

  auto cmd3 = create_apply(2, udf_manager, "matrix_mult",  {1, 4}, {10}, {});   // C_0(1, 0) = A(1, 0) * B(0, 0)
  auto cmd4 = create_apply(3, udf_manager, "matrix_mult",  {3, 5}, {11}, {});   // C_1(1, 0) = A(1, 1) * B(1, 0)

  auto cmd5 = create_apply(4, udf_manager, "matrix_mult",  {0, 6}, {12}, {});   // C_0(0, 1) = A(0, 0) * B(0, 1)
  auto cmd6 = create_apply(5, udf_manager, "matrix_mult",  {2, 7}, {13}, {});   // C_1(0, 1) = A(0, 1) * B(1, 1)

  auto cmd7 = create_apply(6, udf_manager, "matrix_mult",  {1, 6}, {14}, {});   // C_0(1, 1) = A(1, 0) * B(0, 1)
  auto cmd8 = create_apply(7, udf_manager, "matrix_mult",  {3, 7}, {15}, {});   // C_1(1, 1) = A(1, 1) * B(1, 1)

  auto cmd9  = create_reduce(8,  udf_manager, "matrix_add", {8, 9}, 16, {});    // C(0, 0) = C_0(0, 0) + C_1(0, 0)
  auto cmd10 = create_reduce(9,  udf_manager, "matrix_add", {10, 11}, 17, {});  // C(1, 0) = C_0(1, 0) + C_1(1, 0)
  auto cmd11 = create_reduce(10, udf_manager, "matrix_add", {12, 13}, 18, {});  // C(0, 1) = C_0(0, 1) + C_1(0, 1)
  auto cmd12 = create_reduce(11, udf_manager, "matrix_add", {14, 15}, 19, {});  // C(1, 1) = C_0(1, 1) + C_1(1, 1)

  auto cmd13 = create_delete(12, {8, 9, 10, 11, 12, 13, 14, 15}); // remove the intermedite results

  // move them to a vector and schedule them all
  std::vector<bbts::command_ptr_t> to_schedule(13);
  to_schedule[0] = std::move(cmd1);
  to_schedule[1] = std::move(cmd2);
  to_schedule[2] = std::move(cmd3);
  to_schedule[3] = std::move(cmd4);
  to_schedule[4] = std::move(cmd5);
  to_schedule[5] = std::move(cmd6);
  to_schedule[6] = std::move(cmd7);
  to_schedule[7] = std::move(cmd8);
  to_schedule[8] = std::move(cmd9);
  to_schedule[9] = std::move(cmd10);
  to_schedule[10] = std::move(cmd11);
  to_schedule[11] = std::move(cmd12);
  to_schedule[12] = std::move(cmd13);

  scheduler->schedule(to_schedule);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  std::vector<float> check_vals = {1100.0f, 1600.0f, 1900.0f, 2800.0f};
  for(auto val_idx = 0; val_idx < check_vals.size(); ++val_idx) {
    storage->local_transaction(
      {16 + val_idx}, {}, [&](const bbts::storage_t::reservation_result_t &res) {
        auto ts = res.get[0].get().tensor;
        auto &t = ts->as<bbts::dense_tensor_t>();
        for (auto idx = 0; idx < 100 * 100; ++idx) {
          EXPECT_NEAR(t.data()[idx], check_vals[val_idx], 0.001);
        }
    });
  }
}

using matrix_index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<bbts::tid_t, float>>;
using matrix_reduce_index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<float, std::vector<bbts::tid_t>>>;


void init_blocked_matrix(float &val,
                         bbts::tid_t &cur_idx,
                         matrix_index_t &index, 
                         const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                         const bbts::tensor_factory_ptr_t &factory,
                         const bbts::storage_ptr_t &storage,
                         size_t matrix_blocking, 
                         size_t matrix_block_size) {
  
  for(auto row_idx = 0; row_idx < matrix_blocking; ++row_idx) {
    for(auto col_idx = 0; col_idx < matrix_blocking; ++col_idx) {
      
      init_tensor_on_cpu(scheduler, factory, storage, cur_idx,
                         matrix_block_size, matrix_block_size, val);
      std::get<0>(index[{row_idx, col_idx}]) = cur_idx++;
      std::get<1>(index[{row_idx, col_idx}]) = val;
      val += 1.0f;
    }
  }
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
        std::get<0>(reduce_idx[{i, j}]) += std::get<0>(a_index[{i, k}]) * 
                                           std::get<0>(b_index[{k, j}]) *
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

TEST(TestGPUScheduler, Test4) {

  float cur_val = 0.0f;
  bbts::tid_t cur_tid = 0;
  const size_t matrix_size = 200;
  const size_t matrix_blocking = 2;
  const size_t matrix_block_size = matrix_size / matrix_blocking;

  // make the storage
  auto config = std::make_shared<bbts::node_config_t>(0, nullptr);
  config->is_dev_cluster = true;

  auto storage = std::make_shared<bbts::storage_t>(nullptr, config);

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udf_manager = std::make_shared<bbts::udf_manager_t>(factory, nullptr);

  // make the scheduler
  auto scheduler = std::make_shared<bbts::multi_gpu_scheduler_t>(
      4, 128lu * 1024lu * 1024lu, storage, udf_manager, factory);

  // run all the scheduler threads
  auto scheduler_threads = run_threads(scheduler);

  // create tensors on the CPU for matrix A
  matrix_index_t a_index;
  init_blocked_matrix(cur_val, cur_tid, a_index, 
                      scheduler, factory, storage,
                      matrix_blocking, matrix_block_size);

  // create four tensors on the CPU for matrix B
  matrix_index_t b_index;
  init_blocked_matrix(cur_val, cur_tid, b_index, 
                      scheduler, factory, storage,
                      matrix_blocking, matrix_block_size);

  // move them to a vector and schedule them all
  matrix_index_t c_index;
  std::vector<bbts::command_ptr_t> to_schedule = make_multiply(cur_tid,
                                                               udf_manager,
                                                               a_index, 
                                                               b_index, 
                                                               c_index,
                                                               matrix_blocking, 
                                                               matrix_block_size);

  scheduler->schedule(to_schedule);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  for(auto &c_blk : c_index) {
    bbts::tid_t tid = std::get<0>(c_blk.second);
    float value = std::get<1>(c_blk.second);
    storage->local_transaction(
      {tid}, {}, [value](const bbts::storage_t::reservation_result_t &res) {
        auto ts = res.get[0].get().tensor;
        auto &t = ts->as<bbts::dense_tensor_t>();
        for (auto idx = 0; idx < 100 * 100; ++idx) {
          EXPECT_NEAR(t.data()[idx], value, 0.001);
        }
    });
  }
}