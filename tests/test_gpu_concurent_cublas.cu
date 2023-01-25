#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <ostream>
#include <thread>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <mutex>

#include "../main/ud_functions/udf_manager.h"
#include "../main/ud_functions/gpu_scheduler.h"

int main() {

  const int N = 20000;
  const int split = 4;

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // kick off the scheduler
  bbts::gpu_scheduler_ptr_t scheduler = std::make_shared<bbts::gpu_scheduler_t>(factory);
  std::thread sch = std::thread([&](){
    scheduler->run();
  });

  // crate the udf manager
  bbts::udf_manager_t manager(factory, scheduler);

  // return me that matcher for matrix addition
  auto matcher = manager.get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, true);

  // get the impl_id
  auto id = factory->get_tensor_ftm("dense");

  // make the meta
  bbts::dense_tensor_meta_t dm{id, N, N};
  auto &m = dm.as<bbts::tensor_meta_t>();

  // get the number of bytes each matrix needs
  auto num_bytes = factory->get_tensor_size(m);

  bbts::tensor_t* a[split]; 
  checkCudaErrors(cudaMallocManaged(&a[0], num_bytes)); new (a[0]) bbts::tensor_t();
  checkCudaErrors(cudaMallocManaged(&a[1], num_bytes)); new (a[1]) bbts::tensor_t();
  checkCudaErrors(cudaMallocManaged(&a[2], num_bytes)); new (a[2]) bbts::tensor_t();
  checkCudaErrors(cudaMallocManaged(&a[3], num_bytes)); new (a[3]) bbts::tensor_t();

  bbts::tensor_t* b[split][split];
  for(int r = 0; r < split; r++) {
    for(int c = 0; c < split; c++) {
      checkCudaErrors(cudaMallocManaged(&b[r][c], num_bytes));
    }
  }

  for(int c = 0; c < split; c++) {
    for(int r = 0; r < split; r++) {
      auto &me = b[r][c]->as<bbts::dense_tensor_t>();
      factory->init_tensor(&me, m).as<bbts::dense_tensor_t>();
      for(auto i = 0; i < N * N; ++i) {
        me.data()[i] = (float) (i + r * split + c);
      }
    }

    auto &me = a[c]->as<bbts::dense_tensor_t>();
    factory->init_tensor(&me, m).as<bbts::dense_tensor_t>();
    for(auto i = 0; i < N * N; ++i) {
      me.data()[i] = (float) (i + 1);
    }
  }

  std::vector<std::tuple<int, int>> muls;
  for(int t = 0; t < split; t++) {
    for(int c = 0; c < split; c++) {
      muls.push_back({t, c});
    }
  }

  std::random_shuffle ( muls.begin(), muls.end() );

  // run the multiplies
  std::vector<std::thread> threads; threads.reserve(muls.size());
  // kick off the thread
  threads.push_back(std::thread([&, muls]() {
    
    std::vector<bbts::ud_impl_t::tensor_args_t> ins; ins.reserve(muls.size());
    std::vector<bbts::ud_impl_t::tensor_args_t> outs; outs.reserve(muls.size());
    std::vector<bbts::ud_impl_t::tensor_params_t> params; params.reserve(muls.size());
    std::vector<std::future<bool>> futs; futs.reserve(muls.size());

    for(auto mul : muls) {

        // get the indices
        auto idx1 = std::get<0>(mul);
        auto idx2 = std::get<1>(mul);

        // get the tensors a and b
        auto t_a = a[idx1];
        auto t_b = b[idx1][idx2];

        // init the output tensor c
        float *c; checkCudaErrors(cudaMallocManaged(&c, num_bytes));
        bbts::tensor_t &t_c = *((bbts::tensor_t*) c);

        // create the inputs and outputs
        ins.push_back({{t_a, t_b}});
        outs.push_back({{&t_c}});
        params.push_back({});

        // run the kernel
        futs.push_back(scheduler->execute_kernel(ud, &params.back(), &ins.back(), &outs.back()));
    }

    // sync
    for(auto &f : futs) {
      f.get();
    }

  }));

  // wait for it to join
  for(auto &t : threads) {
    t.join();
  }

  // shutdown the scheduler
  scheduler->shutdown();

  // wait
  sch.join();

  return 0;
}