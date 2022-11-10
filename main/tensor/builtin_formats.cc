#include "builtin_formats.h"
#include "tensor.h"
#include <iostream>
#include <sstream>

namespace bbts {

tensor_creation_fs_t bbts::dense_tensor_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(dense_tensor_t *) here;
    auto &m = *(dense_tensor_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    auto &m = *(dense_tensor_meta_t *) &_meta;
    return sizeof(tensor_t) + (size_t) m.m().num_cols * (size_t) m.m().num_rows * sizeof(float);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(dense_tensor_t *) here;

    // extract the info
    auto num_rows = t.meta().m().num_rows;
    auto num_cols = t.meta().m().num_cols;
    auto data = t.data();

    // print the tensor
    for(int i = 0; i < num_rows; i++) {
      ss << "[";
      for(int j = 0; j < num_cols; j++) {
        ss << data[i * num_cols + j] << ((j == num_cols - 1) ? "" : ",");
      }
      ss << "]\n";
    }

  };

  // return the tensor creation functions
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
}

}