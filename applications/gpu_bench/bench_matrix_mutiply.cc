#include "../../src/gpu/scheduler.h"
#include "../../src/tensor/builtin_formats.h"
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <utility>
#include <vector>

#include "mm_util.h"

using namespace std::chrono;

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


int main() {

  const int32_t num_gpus = 4;
  float cur_val = 0.0f;
  bbts::tid_t cur_tid = 0;
  const size_t matrix_size = 8000;
  const size_t matrix_blocking = 4;
  const size_t matrix_block_size = matrix_size / matrix_blocking;

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

  // create tensors on the CPU for matrix A
  matrix_index_t a_index;
  init_blocked_matrix(cur_val, cur_tid, a_index, 
                      scheduler, factory, storage,
                      matrix_blocking, matrix_block_size,
                      matrix_blocking, matrix_block_size);

  // create four tensors on the CPU for matrix B
  matrix_index_t b_index;
  init_blocked_matrix(cur_val, cur_tid, b_index, 
                      scheduler, factory, storage,
                      matrix_blocking, matrix_block_size,
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
  sleep(2);

  scheduler->schedule(to_schedule);

  // move all the tensors currently in the GPU back into RAM
  scheduler->flush();

  // finish all the threads
  scheduler->shutdown();
  for (auto &t : scheduler_threads) {
    t.join();
  }

  scheduler->save_log("gpu.proto");

  // for(auto &c_blk : c_index) {
  //   bbts::tid_t tid = std::get<0>(c_blk.second);
  //   float value = std::get<1>(c_blk.second);
  //   storage->local_transaction(
  //     {tid}, {}, [value](const bbts::storage_t::reservation_result_t &res) {
  //       auto ts = res.get[0].get().tensor;
  //       auto &t = ts->as<bbts::dense_tensor_t>();
  //       for (auto idx = 0; idx < 100 * 100; ++idx) {
  //       //   EXPECT_NEAR(t.data()[idx], value, 0.1f);
  //       }
  //   });
  // }
}