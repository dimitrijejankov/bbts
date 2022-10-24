#include "../main/tensor/builtin_formats.h"
#include "../main/ud_functions/builtin_functions.h"
#include "../main/ud_functions/udf_manager.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace bbts;

int main() {

  // create the tensor factory
  auto factory = std::make_shared<tensor_factory_t>();

  // get the impl_id
  auto id = factory->get_tensor_ftm("dense");

  // crate the udf manager
  udf_manager_t manager(factory, nullptr);

  // figure out the sum
  auto matcher = manager.get_matcher_for("matrix_add");
  auto add = matcher->findMatch({"dense", "dense"}, {"dense"}, false);

  // figure out the multiply
  matcher = manager.get_matcher_for("matrix_mult");
  auto mult = matcher->findMatch({"dense", "dense"}, {"dense"}, false);

  float network_speed_Bps = 1.25 / (1024 * 1024 * 1024);
  const size_t num_iter = 10;

  for (auto size : std::vector<uint32_t>{400, 800, 1600, 3200}) {

    // make the meta
    dense_tensor_meta_t dm{id, size, size};
    auto &m = dm.as<tensor_meta_t>();

    // get how much we need to allocate
    auto num_bytes = factory->get_tensor_size(m);

    // the memory
    std::unique_ptr<char[]> a_mem(new char[num_bytes]); new (a_mem.get()) tensor_t();
    std::unique_ptr<char[]> b_mem(new char[num_bytes]); new (b_mem.get()) tensor_t();
    std::unique_ptr<char[]> c_mem(new char[num_bytes]); new (c_mem.get()) tensor_t();

    // init the two tensors
    auto &a =
        factory->init_tensor((tensor_t *)a_mem.get(), m).as<dense_tensor_t>();
    auto &b =
        factory->init_tensor((tensor_t *)b_mem.get(), m).as<dense_tensor_t>();
    auto &c =
        factory->init_tensor((tensor_t *)b_mem.get(), m).as<dense_tensor_t>();

    // write some values to a
    auto am = a.meta().m();
    for (auto row = 0; row < am.num_rows; ++row) {
      for (auto col = 0; col < am.num_cols; ++col) {
        a.data()[row * am.num_cols + col] = float(row + col);
      }
    }

    // write some values to b
    auto bm = b.meta().m();
    for (auto row = 0; row < bm.num_rows; ++row) {
      for (auto col = 0; col < bm.num_cols; ++col) {
        b.data()[row * bm.num_cols + col] = 2.0f * float(row + col);
      }
    }

    // write zeros values to b
    auto cm = b.meta().m();
    for (auto row = 0; row < cm.num_rows; ++row) {
      for (auto col = 0; col < cm.num_cols; ++col) {
        b.data()[row * cm.num_cols + col] = 0.0f;
      }
    }

    ud_impl_t::tensor_args_t input_args = {{&a, &b}};
    ud_impl_t::tensor_args_t output_args = {{&c}};

    // call the addition
    auto add_start = high_resolution_clock::now();
    for (int32_t num_iters = 0; num_iters < num_iter; ++num_iters) {
      add->call_ud({._params = bbts::command_param_list_t{._data = nullptr,
                                                          ._num_elements = 0}, ._additional_memory = nullptr},
                   input_args, output_args);
    }
    auto add_stop = high_resolution_clock::now();
    auto add_duration = duration_cast<microseconds>(add_stop - add_start);


    // call the mult
    auto mult_start = high_resolution_clock::now();
    for (int32_t num_iters = 0; num_iters < num_iter; ++num_iters) {
      mult->call_ud({._params = bbts::command_param_list_t{._data = nullptr,
                                                          ._num_elements = 0}, ._additional_memory = nullptr},
                   input_args, output_args);
    }
    auto mult_stop = high_resolution_clock::now();
    auto mult_duration = duration_cast<microseconds>(mult_stop - mult_start);

    auto add_time = (float) add_duration.count() / (float) (size * size * num_iter * 1000000.0f);
    auto matrix_time = (float) mult_duration.count() / (float) (size * size * size * num_iter * 1000000.0f);

    std::cout << "add_const : " << add_time << ", " << "mult_const : " << matrix_time << '\n';
  }


  return 0;
}
