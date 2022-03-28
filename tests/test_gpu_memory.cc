#include "../src/gpu/gpu_memory.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using namespace bbts;

command_id_t command_id = 0;

kernel_prep_ptr_t make_kernel_prep(int32_t dev, 
                                   const std::vector<tid_t> &inputs,
                                   const std::vector<size_t> &input_sizes,
                                   const std::vector<tid_t> &outputs,
                                   const std::vector<size_t> &output_sizes) {

  // fill out the kernel prep
  auto kp = std::make_shared<kernel_prep_t>();
  kp->command_id = command_id++;
  kp->cpu_done = false;
  kp->gpu_done = false;
  kp->cpu_transfers = {};
  kp->gpu_transfers = {};
  kp->dev = 0;
  kp->input = inputs;
  kp->input_sizes = input_sizes;
  kp->output = outputs;
  kp->output_sizes = output_sizes;
  kp->type = bbts::command_t::APPLY;
  
  // make the kernel run
  kp->run_me = std::make_shared<kernel_run_t>();
  kp->run_me->ud = nullptr;
  kp->run_me->inputs.resize(inputs.size());
  kp->run_me->outputs.resize(outputs.size());
  kp->run_me->params = {};

  // reeturn the kernel prep
  return std::move(kp);
}


TEST(TestGPUMemory, TestCanPreallocate1) {

  gpu_memory_t gpu_memory(4, 1024);

  auto prep = make_kernel_prep(0, {0, 1}, {256, 256}, {2}, {256});
  
  gpu_memory.tensor_loaded_on_cpu(0, 256);
  gpu_memory.tensor_loaded_on_cpu(1, 256);

  EXPECT_TRUE(gpu_memory.can_preallocate(prep) != -1);
}

TEST(TestGPUMemory, TestCanPreallocate2) {

  gpu_memory_t gpu_memory(4, 1024);

  auto prep = make_kernel_prep(0, {0, 1}, {512, 512}, {2}, {512});
  
  gpu_memory.tensor_loaded_on_cpu(0, 512);
  gpu_memory.tensor_loaded_on_cpu(1, 512);

  EXPECT_TRUE(gpu_memory.can_preallocate(prep) == -1);
}

TEST(TestGPUMemory, TestPreallocate) {

  gpu_memory_t gpu_memory(1, 1024);

  auto prep1 = make_kernel_prep(0, {0, 1}, {256, 128}, {2}, {128});
  auto prep2 = make_kernel_prep(0, {2, 3}, {128, 256}, {4}, {128});
  
  gpu_memory.tensor_loaded_on_cpu(0, 256);
  gpu_memory.tensor_loaded_on_cpu(1, 128);
  gpu_memory.tensor_loaded_on_cpu(3, 256);

  auto dev1 = gpu_memory.can_preallocate(prep1);
  EXPECT_TRUE(dev1 != -1);

  // allocate the frist one
  gpu_memory.preallocate(prep1, dev1);

  // tensors 0 and 1 should be loaded
  auto &t1 = prep1->cpu_transfers[0];
  auto &t2 = prep1->cpu_transfers[1];

  // check if we have the right number of transfers
  EXPECT_EQ(prep1->cpu_transfers.size(), 2);
  EXPECT_EQ(prep1->gpu_transfers.size(), 0);

  // make sure the transfers are properly initalized
  EXPECT_TRUE(t1->dst != nullptr);
  EXPECT_EQ(t1->dst_dev, 0);
  EXPECT_GE(t1->id, 0);
  EXPECT_EQ(t1->is_finished, false);
  EXPECT_EQ(t1->num_bytes, 256);
  EXPECT_EQ(t1->tid, 0);

  EXPECT_TRUE(t2->dst != nullptr);
  EXPECT_EQ(t2->dst_dev, 0);
  EXPECT_GE(t2->id, 0);
  EXPECT_EQ(t2->is_finished, false);
  EXPECT_EQ(t2->num_bytes, 128);
  EXPECT_EQ(t2->tid, 1);

  // mark the transfers as finished
  t1->is_finished = true;
  t2->is_finished = true;

  // finish the kernel prep1
  gpu_memory.finish_kernel_prep(prep1);

  // preallocate here
  auto dev2 = gpu_memory.can_preallocate(prep2);
  EXPECT_TRUE(dev2 != -1);

  // allocate the second one
  gpu_memory.preallocate(prep2, dev2);

  // check if we have the right number of transfers
  EXPECT_EQ(prep2->cpu_transfers.size(), 1);
  EXPECT_EQ(prep2->gpu_transfers.size(), 0);

  // make sure the transfers are properly initalized
  auto &t3 = prep2->cpu_transfers[0];
  EXPECT_TRUE(t3->dst != nullptr);
  EXPECT_EQ(t3->dst_dev, 0);
  EXPECT_GE(t3->id, 0);
  EXPECT_EQ(t3->is_finished, false);
  EXPECT_EQ(t3->num_bytes, 256);
  EXPECT_EQ(t3->tid, 3);

  // finish the kernel prep2
  gpu_memory.finish_kernel_prep(prep2);
}
