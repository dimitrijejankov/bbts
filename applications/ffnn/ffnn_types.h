#pragma once

#include "../../main/tensor/tensor.h"
#include <cstdint>

namespace bbts {

struct ffnn_dense_meta_t : public tensor_meta_t {

  // the meta stucture
  struct m_t {

    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t row_idx;
    uint32_t col_idx;
    bool has_bias;
    uint32_t num_aggregated;
  };

  // returns the meta data struct
  m_t &m() const {

    // we use it as the blob
    return *((m_t *)_blob);
  }

  // init the tensor with the format impl_id
  ffnn_dense_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta with row and column numbers
  ffnn_dense_meta_t(tfid_t _id, bool has_bias, uint32_t num_rows,
                    uint32_t row_idx, uint32_t col_idx, uint32_t num_cols)
      : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_rows = num_rows,
                 .num_cols = num_cols,
                 .has_bias = has_bias,
                 .num_aggregated = 1};
  }
};

struct ffnn_dense_t : public tensor_t {

  // return the meta data of the dense tensor
  ffnn_dense_meta_t &meta() const { return get_meta<ffnn_dense_meta_t>(); }

  // return the
  float *data() const { return &get_data<float>(); }

  // returns the bias
  float *bias() const {
    return (data() + meta().m().num_rows * meta().m().num_cols);
  }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs();
};

struct ffnn_sparse_meta_t : public tensor_meta_t {

  // the meta stucture
  struct m_t {

    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t nnz;
  };

  // returns the meta data struct
  m_t &m() const {

    // we use it as the blob
    return *((m_t *)_blob);
  }

  // init the tensor with the format impl_id
  ffnn_sparse_meta_t(tfid_t _id) : tensor_meta_t{.fmt_id = _id} {}

  // init the tensor meta with row and column numbers
  ffnn_sparse_meta_t(tfid_t _id, uint32_t num_rows, uint32_t num_cols,
                     uint32_t nnz)
      : tensor_meta_t{.fmt_id = _id} {
    this->m() = {.num_rows = num_rows, .num_cols = num_cols, .nnz = nnz};
  }
};

// layout nnz * {row, col, val}
struct ffnn_sparse_t : public tensor_t {

  // the value
  struct ffnn_sparse_value_t {

    uint32_t row;
    uint32_t col;
    float val;
  };

  // return the meta data of the dense tensor
  ffnn_sparse_meta_t &meta() const { return get_meta<ffnn_sparse_meta_t>(); }

  // return the
  ffnn_sparse_value_t *data() const { return &get_data<ffnn_sparse_value_t>(); }

  // return creation functions
  static tensor_creation_fs_t get_creation_fs();
};

} // namespace bbts