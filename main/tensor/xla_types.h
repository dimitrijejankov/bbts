#pragma once

#ifdef BBTS_BAZEL_BUILD

#include "../../main/tensor/tensor.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include <cstdint>

namespace bbts {

struct xla_meta_t : public tensor_meta_t {

  // the meta stucture
  struct m_t {
    size_t size;
  };

  // returns the meta data struct
  m_t &m() const {

    // we use it as the blob
    return *((m_t *)_blob);
  }

  // init the tensor with the format impl_id
  xla_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta with row and column numbers
  xla_meta_t(tfid_t _id, size_t _size) : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.size = _size};
  }
};

struct xla_tensor_t : public tensor_t {

  // return the meta data of the dense tensor
  xla_meta_t &meta() const { return get_meta<xla_meta_t>(); }

  // return the
  float *data() const { return &get_data<float>(); }

  // return it as tensorflow DeviceMemoryBase
  tensorflow::se::DeviceMemoryBase as_tf_data() const { 
    return tensorflow::se::DeviceMemoryBase(data(), sizeof(float) * meta().m().size);
  };

  // return creation functions
  static tensor_creation_fs_t get_creation_fs();
};

}
#endif