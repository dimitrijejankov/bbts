#include "../src/gpu/scheduler.h"
#include <gtest/gtest.h>

using namespace std::chrono;

bbts::command_ptr_t create_apply(bbts::udf_manager_ptr udm,
                                 const std::string &ud_name,
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
  auto ud = matcher->findMatch({}, {"dense"}, true);
  auto cmd = bbts::command_t::create_apply(0, ud->impl_id, true, 
                                           params, prep_in, prep_out);
  return std::move(cmd);
}

std::vector<std::thread>
run_threads(bbts::multi_gpu_scheduler_ptr_t scheduler) {

  std::vector<std::thread> threads;
  threads.push_back(
      std::thread([scheduler]() { scheduler->command_prep_thread(); }));

  threads.push_back(
      std::thread([scheduler]() { scheduler->cpu_to_gpu_thread(); }));

  for (auto dev = 0; dev < scheduler->_num_gpus; ++dev) {

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
  scheduler->schedule_apply(std::move(create_apply(udf_manager, "const", {}, {0}, { bbts::command_param_t{.i = 100}, 
                                                                                    bbts::command_param_t{.i = 100},
                                                                                    bbts::command_param_t{.f = 2.0f}} )));

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  // check the created tensor
  storage->local_transaction(
    {0}, {},
    [&](const bbts::storage_t::reservation_result_t &res) {
      auto ts = res.get[0].get().tensor;
      auto &t = ts->as<bbts::dense_tensor_t>();
      for(auto idx = 0; idx < 100 * 100; ++idx) {
        EXPECT_NEAR(t.data()[idx], 2.0f, 0.001); 
      }
    });
}