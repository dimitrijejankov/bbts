#include <iostream>
#include <memory>
#include <vector>
#include <sstream>

#include <algorithm> // fill
#include <numeric>   // iota.

#include "../../main/ud_functions/udf_manager.h"
#include "xla_types.h"
#include "../../main/ud_functions/xla_kernel_base.h"
#include "xla_test_kernel.h"

int main() {

  /// 1. Register the format and the kernel

  // create the tensor factory
  auto factory = std::make_shared<bbts::tensor_factory_t>();

  // register the format
  factory->register_fmt("xla_dense", bbts::xla_tensor_t::get_creation_fs());

  // crate the udf manager
  bbts::udf_manager_t manager(factory, nullptr);

  // register the ud fucntion
  manager.register_udf(std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
        .ud_name = "xla_test_kernel",
        .is_ass = false,
        .is_com = false,
        .num_in = 2,
        .num_out = 1,
        .impls = {}
      }));
  manager.register_udf_impl(std::make_unique<bbts::xla_test_kernel>());


  /// 2. Do the prep

  // return me that matcher for matrix addition
  auto matcher = manager.get_matcher_for("xla_test_kernel");

  // get the ud object
  auto ud = matcher->findMatch({"xla_dense", "xla_dense"}, {"xla_dense"}, false);

  // get the tensor format
  auto id = factory->get_tensor_ftm("xla_dense");

  // we only cast this so I can grab the size
  auto xla_ud = dynamic_cast<bbts::xla_kernel_base*>(ud);

  bbts::xla_meta_t a(id, xla_ud->input_sizes[0]);
  auto &m_a = a.as<bbts::tensor_meta_t>();

  bbts::xla_meta_t b(id, xla_ud->input_sizes[1]);
  auto &m_b = b.as<bbts::tensor_meta_t>();
  
  bbts::xla_meta_t c(id, xla_ud->output_sizes[0]);
  auto &m_c = c.as<bbts::tensor_meta_t>();

  bbts::ud_impl_t::meta_args_t meta_input_args = {{&m_a, &m_b}};
  bbts::ud_impl_t::meta_args_t meta_output_args = {{&m_c}};

  size_t extra_size = ud->get_required_memory({}, meta_input_args);

  // get the extra memory
  std::cout << "Num Inputs : " << xla_ud->input_sizes.size() << ", Num Outputs : " << xla_ud->output_sizes.size() << "\n";
  std::cout << "Extra memory required : " << extra_size << "\n";
  std::cout << "a size : " << xla_ud->input_sizes[0] << "\n";
  std::cout << "b size : " << xla_ud->input_sizes[1] << "\n";
  std::cout << "c size : " << xla_ud->output_sizes[0] << "\n";
  
  auto ma_size = factory->get_tensor_size(m_a);
  auto mb_size = factory->get_tensor_size(m_b);
  auto mc_size = factory->get_tensor_size(m_c);

  std::unique_ptr<char[]> a_mem(new char[ma_size]); new (a_mem.get()) bbts::tensor_t();
  std::unique_ptr<char[]> b_mem(new char[mb_size]); new (b_mem.get()) bbts::tensor_t();
  std::unique_ptr<char[]> c_mem(new char[mc_size]); new (c_mem.get()) bbts::tensor_t();
  std::unique_ptr<char[]> tmp_mem(new char[extra_size]);

  auto &aa = factory->init_tensor((bbts::tensor_t*) a_mem.get(), m_a).as<bbts::xla_tensor_t>();
  auto &bb = factory->init_tensor((bbts::tensor_t*) b_mem.get(), m_b).as<bbts::xla_tensor_t>();
  auto &cc = factory->init_tensor((bbts::tensor_t*) c_mem.get(), m_c).as<bbts::xla_tensor_t>();

  // set some values
  std::fill(aa.data(), aa.data() + (xla_ud->input_sizes[0] / sizeof(float)), 1.0f);
  std::fill(bb.data(), bb.data() + (xla_ud->input_sizes[1] / sizeof(float)), 1.0f);
  std::iota(cc.data(), cc.data() + (xla_ud->output_sizes[0] / sizeof(float)), 0.0f);

  // form the inputs
  bbts::ud_impl_t::tensor_args_t input_args = {{&aa, &bb}};
  bbts::ud_impl_t::tensor_args_t output_args = {{&cc}};

  // call the addition
  ud->call_ud({ ._params = bbts::command_param_list_t {._data = nullptr, ._num_elements = 0}, ._additional_memory = tmp_mem.get() }, input_args, output_args);

  std::stringstream ss;
  factory->print_tensor(&cc, ss);

  std::cout << ss.str() << "\n";
}

