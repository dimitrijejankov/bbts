
#include "../../main/tensor/tensor_factory.h"
#include "../../main/ud_functions/udf_manager.h"

#include "xla_types.h"  
#include "xla_test_kernel.h"  


extern "C" {

  void register_tensors(bbts::tensor_factory_ptr_t tensor_factory) {
    tensor_factory->register_fmt("xla_dense", bbts::xla_tensor_t::get_creation_fs());
  }
 
  void register_udfs(bbts::udf_manager_ptr udf_manager) {
    
    // register the ud fucntion
    udf_manager->register_udf(std::make_unique<bbts::ud_func_t>(
        bbts::ud_func_t {
          .ud_name = "xla_test_kernel",
          .is_ass = false,
          .is_com = false,
          .num_in = 2,
          .num_out = 1,
          .impls = {}
        }));
    udf_manager->register_udf_impl(std::make_unique<bbts::xla_test_kernel>());
  }
}
