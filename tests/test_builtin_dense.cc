#include <gtest/gtest.h>
#include "../src/tensor/tensor_factory.h"
#include "../src/tensor/builtin_formats.h"

namespace bbts {

TEST(TestDenseTensor, FormatRegisted) {

  // create the tensor factory
  tensor_factory_t factory;

  // get the impl_id
  auto id = factory.get_tensor_ftm("dense");

  // it has to find the format
  EXPECT_NE(id, -1);
}

TEST(TestDenseTensor, GetSize) {

  // create the tensor factory
  tensor_factory_t factory;

  // get the impl_id
  auto id = factory.get_tensor_ftm("dense");

  // check if get size works
  for (uint num_rows = 0; num_rows < 100; num_rows++) {
    for (uint num_cols = 0; num_cols < 100; num_cols++) {

      // make the meta
      dense_tensor_meta_t dm{id, num_rows, num_cols};
      auto &m = *((tensor_meta_t *) &dm);

      // we expect the size to be sizeof(tensor_meta_t) + 10 * 10 * sizeof(float)
      EXPECT_EQ(sizeof(tensor_t) + dm.m().num_rows * dm.m().num_cols * sizeof(float), factory.get_tensor_size(m));
    }
  }
}

TEST(TestDenseTensor, Init) {

  // create the tensor factory
  tensor_factory_t factory;

  // get the impl_id
  auto id = factory.get_tensor_ftm("dense");

  // make the meta
  dense_tensor_meta_t dm{id, 100, 200};
  auto &m = dm.as<tensor_meta_t>();

  // get how much we need to allocate
  auto size = factory.get_tensor_size(m);

  // the memory
  std::unique_ptr<char[]> a_mem(new char[size]); new (a_mem.get()) tensor_t();
  std::unique_ptr<char[]> b_mem(new char[size]); new (b_mem.get()) tensor_t();

  // init the two tensors
  auto &a = factory.init_tensor((tensor_t*) a_mem.get(), m).as<dense_tensor_t>();
  auto &b = factory.init_tensor((tensor_t*) b_mem.get(), m).as<dense_tensor_t>();

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
    for(auto col = 0; col < bm.num_cols; ++col) {
      b.data()[row * bm.num_cols + col] =  2.0f * float(row + col);
    }
  }

  // add b to a
  for(auto row = 0; row < am.num_rows; ++row) {
    for(auto col = 0; col < am.num_cols; ++col) {
      a.data()[row * am.num_cols + col] += b.data()[row * am.num_cols + col];
    }
  }

  // check that the values are correct
  for(auto row = 0; row < am.num_rows; ++row) {
    for (auto col = 0; col < am.num_cols; ++col) {
      EXPECT_LE(std::abs(a.data()[row * am.num_cols + col] - 3.0f * float(row + col)), 0.0001f);
    }
  }

  // make sure the size is correct
  EXPECT_EQ(am.num_rows, 100);
  EXPECT_EQ(am.num_cols, 200);
  EXPECT_EQ(bm.num_rows, 100);
  EXPECT_EQ(bm.num_cols, 200);
}

}