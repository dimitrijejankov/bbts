#include "../src/gpu/scheduler.h"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

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
  scheduler->schedule_apply(std::move(create_apply(
      0, udf_manager, "const", {}, {0},
      {bbts::command_param_t{.i = 100}, bbts::command_param_t{.i = 100},
       bbts::command_param_t{.f = 2.0f}})));

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
  auto cmd = create_reduce(0, udf_manager, "matrix_add", 
                           {0, 1, 2, 3}, 4, {});
  scheduler->schedule_reduce(std::move(cmd));

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