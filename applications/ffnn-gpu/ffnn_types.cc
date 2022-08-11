#include "ffnn_types.h"
#include <iostream>
#include <sstream>

namespace bbts {

tensor_creation_fs_t bbts::ffnn_dense_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(ffnn_dense_t *) here;
    auto &m = *(ffnn_dense_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    auto &m = *(ffnn_dense_meta_t *) &_meta;

    auto num_elements = m.m().num_cols * m.m().num_rows;
    num_elements += m.m().has_bias ? m.m().num_cols : 0;

    // we add the bias if we have it to the size
    return sizeof(tensor_t) + num_elements * sizeof(float);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(ffnn_dense_t *) here;

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


tensor_creation_fs_t bbts::ffnn_sparse_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(ffnn_sparse_t *) here;
    auto &m = *(ffnn_sparse_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    
    // we add the bias if we have it to the size
    auto &m = *(ffnn_sparse_meta_t *) &_meta;
    return sizeof(tensor_t) + m.m().nnz * sizeof(ffnn_sparse_t::ffnn_sparse_value_t);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(ffnn_sparse_t *) here;

    // extract the info
    auto num_rows = t.meta().m().num_rows;
    auto num_cols = t.meta().m().num_cols;
    auto nnz = t.meta().m().nnz;
    auto data = t.data();

    // print the tensor 
    ss << "{ rows = " << num_rows << ", cols = " << num_cols << " }";
    ss << "[\n";
    for(int i = 0; i < nnz; i++) {
      ss << "(" << data[i].row << ", " << data[i].col << ", " << data[i].val << ")" << ((i == (nnz - 1)) ? "" : ",") << "\n";
    }
    ss << "]\n";

  };

  // return the tensor creation functions
  return tensor_creation_fs_t{.get_size = size, .init_tensor = init, .print = pnt};
}

}