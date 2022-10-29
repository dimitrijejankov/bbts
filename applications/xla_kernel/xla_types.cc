#include "xla_types.h"
#include <iostream>
#include <sstream>

namespace bbts {

tensor_creation_fs_t bbts::xla_tensor_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(xla_tensor_t *) here;
    auto &m = *(xla_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    
    // we add the bias if we have it to the size
    auto &m = *(xla_meta_t *) &_meta;
    return sizeof(tensor_t) + m.m().size;
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(xla_tensor_t *) here;

    // extract the info
    auto data = t.data();

    for(int i = 0; i != t.meta().m().size; ++i) {
      std::cout << data[i] << " ";
    }
    std::cout << std::endl;
  };

  // return the tensor creation functions
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
}

}