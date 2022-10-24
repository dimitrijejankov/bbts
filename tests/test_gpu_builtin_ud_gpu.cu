#include <memory>
#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <thread>

#include "../main/tensor/builtin_formats.h"
#include "../main/ud_functions/builtin_functions.h"
#include "../main/ud_functions/udf_manager.h"

#include "../third_party/cuda/gpu.h"

namespace bbts {

TEST(TestBuiltinMatrix, TestDenseMatrixAdditonInplace) {

  // create the tensor factory
  auto factory = std::make_shared<tensor_factory_t>();

  // kick off the scheduler
  bbts::gpu_scheduler_ptr_t scheduler = std::make_shared<bbts::gpu_scheduler_t>(factory);
  std::thread sch = std::thread([&](){
    scheduler->run();
  });

  // crate the udf manager
  udf_manager_t manager(factory, scheduler);

  // return me that matcher for matrix addition
  auto matcher = manager.get_matcher_for("matrix_add");

  // make sure we got a matcher
  EXPECT_NE(matcher, nullptr);

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, true);

  // check if we actually got it
  EXPECT_NE(ud, nullptr);

  // get the impl_id
  auto id = factory->get_tensor_ftm("dense");

  std::vector<std::thread> threads; threads.reserve(10);
  for(int32_t i = 0; i < 10; i++) {

    // kick off a thread
    threads.emplace_back(std::thread([&]() {

        // make the meta
        dense_tensor_meta_t dm{id, 100, 200};
        auto &m = dm.as<tensor_meta_t>();

        // get how much we need to allocate
        auto size = factory->get_tensor_size(m);

        // the memory
        char *a_mem; cudaMallocManaged(&a_mem, size); new (a_mem) tensor_t();
        char *b_mem; cudaMallocManaged(&b_mem, size); new (b_mem) tensor_t();
        char *c_mem; cudaMallocManaged(&c_mem, size); new (c_mem) tensor_t();

        // init the two tensors
        auto &a = factory->init_tensor((tensor_t*) a_mem, m).as<dense_tensor_t>();
        auto &b = factory->init_tensor((tensor_t*) b_mem, m).as<dense_tensor_t>();
        auto &c = factory->init_tensor((tensor_t*) c_mem, m).as<dense_tensor_t>();

        // write some values to a
        auto am = a.meta().m();
        for(auto row = 0; row < am.num_rows; ++row) {
          for(auto col = 0; col < am.num_cols; ++col) {
            a.data()[row * am.num_cols + col] = float (row + col);
          }
        }

        // write some values to b
        auto bm = b.meta().m();
        for(auto row = 0; row < bm.num_rows; ++row) {
          for (auto col = 0; col < bm.num_cols; ++col) {
            b.data()[row * bm.num_cols + col] = 2.0f * float(row + col);
          }
        }

        ud_impl_t::tensor_args_t input_args = {{&a, &b}};
        ud_impl_t::tensor_args_t output_args = {{&c}};

        // call the addition
        ud->call_ud({ ._params = bbts::command_param_list_t {._data = nullptr, ._num_elements = 0} }, input_args, output_args);

        // sync the device
        auto error = cudaDeviceSynchronize();
        checkCudaErrors(error);
        
        // check that the values are correct
        for(auto row = 0; row < am.num_rows; ++row) {
          for (auto col = 0; col < am.num_cols; ++col) {
            EXPECT_LE(std::abs(c.data()[row * am.num_cols + col] - 3.0f * float(row + col)), 0.0001f);
          }
        }

        // get the meta
        am = a.meta().m();
        bm = b.meta().m();

        // make sure the dimensions of the output are correct and that the other input has not been altered
        EXPECT_EQ(am.num_rows, 100);
        EXPECT_EQ(am.num_cols, 200);
        EXPECT_EQ(bm.num_rows, 100);
        EXPECT_EQ(bm.num_cols, 200);
    }));
  }

  // sync
  for(auto &t : threads) {
    t.join();
  }

  // shutdown the scheduler
  scheduler->shutdown();

  sch.join();

  cudaProfilerStop();
}

}