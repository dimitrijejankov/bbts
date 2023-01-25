#include <gtest/gtest.h>
#include "../main/tensor/tensor_factory.h"
#include "../main/tensor/builtin_formats.h"

using namespace bbts;

static inline bool is_aligned(const void *pointer, size_t byte_count) { 
    std::cout << (uintptr_t)pointer % byte_count << '\n';
    return (uintptr_t)pointer % byte_count == 0; 
}

TEST(TestTensorAligment, Test1) {

    auto ts = (tensor_t*) aligned_alloc(256, 1000);
    new (ts) tensor_t();

    std::cout << "Size of dense_tensor_t: " << sizeof(dense_tensor_t) << '\n';
    std::cout << "Size of tensor_t: " << sizeof(tensor_t) << '\n';

    EXPECT_TRUE(is_aligned((void*) ts, 256));
    EXPECT_TRUE(is_aligned((char*) ts + sizeof(tensor_t), 256));
    EXPECT_TRUE(is_aligned(ts->as<dense_tensor_t>().get_data_ptr<void>(), 256));
}