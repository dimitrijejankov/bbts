#include "../src/gpu/gpu_memory.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using namespace bbts;

command_id_t command_id = 0;

void fake_process_gc_request(gc_request_ptr_t gc) {

  // just free
  for(auto to_free : gc->to_free) {
    auto [mem, tid, num_bytes] = to_free;
    cudaFree(mem);
  }

  // free all since we don't really care about the data
  for(auto to_evict : gc->to_evict) {
    auto [mem, tid, num_bytes] = to_evict;
    cudaFree(mem);
  }
}

apply_schedule_ptr_t create_apply(command_id_t id,
                                  const std::vector<tid_t> &inputs, 
                                  const std::vector<size_t> &inputs_size, 
                                  const std::vector<tid_t> &outputs,
                                  const std::vector<size_t> &outputs_size) {

  std::vector<command_t::tid_node_id_t> in;
  for(auto &i : inputs) {
    in.push_back(command_t::tid_node_id_t{.tid = i, .node = 0});
  }

  std::vector<command_t::tid_node_id_t> out;
  for(auto &o : outputs) {
    out.push_back(command_t::tid_node_id_t{.tid = o, .node = 0});
  }

  // make the actual apply schedule
  auto apply = std::make_shared<bbts::command_schedule_t>(); 
  apply->cmd = command_t::create_apply(id, {0, 0}, true, {}, in, out);

  // we don't care about this
  apply->input_sizes = inputs_size;
  apply->output_sizes = outputs_size;
  apply->fn = nullptr;

  return std::move(apply);
}

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

  EXPECT_TRUE(gpu_memory.can_preallocate(prep, 0) != -1);
}

TEST(TestGPUMemory, TestCanPreallocate2) {

  gpu_memory_t gpu_memory(4, 1024);

  auto prep = make_kernel_prep(0, {0, 1}, {512, 512}, {2}, {512});
  
  gpu_memory.tensor_loaded_on_cpu(0, 512);
  gpu_memory.tensor_loaded_on_cpu(1, 512);

  EXPECT_TRUE(gpu_memory.can_preallocate(prep, 0) == -1);
}

TEST(TestGPUMemory, TestPreallocate) {

  gpu_memory_t gpu_memory(1, 1024);

  auto prep1 = make_kernel_prep(0, {0, 1}, {256, 128}, {2}, {128});
  auto prep2 = make_kernel_prep(0, {2, 3}, {128, 256}, {4}, {128});
  
  gpu_memory.tensor_loaded_on_cpu(0, 256);
  gpu_memory.tensor_loaded_on_cpu(1, 128);
  gpu_memory.tensor_loaded_on_cpu(3, 256);

  auto dev1 = gpu_memory.can_preallocate(prep1, 0);
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
  auto dev2 = gpu_memory.can_preallocate(prep2, 0);
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


TEST(TestGPUMemory, TestGC1) {

  gpu_memory_t gpu_memory(1, 1024);

  // we want to allocate these
  auto prep1 = make_kernel_prep(0, {0, 1}, {256, 256}, {2}, {512});
  auto prep2 = make_kernel_prep(0, {3, 4}, {256, 256}, {5}, {512});

  // mark the apply for use
  gpu_memory.mark_for_use(create_apply(0, {0, 1}, {256, 256}, {2}, {512}));
  gpu_memory.mark_for_use(create_apply(0, {3, 4}, {256, 256}, {5}, {512}));

  // mark that these tensors are loaded on the CPU
  gpu_memory.tensor_loaded_on_cpu(0, 256);
  gpu_memory.tensor_loaded_on_cpu(1, 256);
  gpu_memory.tensor_loaded_on_cpu(3, 256);
  gpu_memory.tensor_loaded_on_cpu(4, 256);

  // make sure 
  auto dev1 = gpu_memory.can_preallocate(prep1, 0);
  EXPECT_EQ(dev1, 0);

  // preallocate the krenel prep
  gpu_memory.preallocate(prep1, 0);

  // check if we have the right number of transfers
  EXPECT_EQ(prep1->cpu_transfers.size(), 2);
  EXPECT_EQ(prep1->gpu_transfers.size(), 0);

  // get the transfers
  auto t1 = prep1->cpu_transfers[0];
  auto t2 = prep1->cpu_transfers[1];

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
  EXPECT_EQ(t2->num_bytes, 256);
  EXPECT_EQ(t2->tid, 1);

  // mark as finished
  t1->is_finished = true;
  t2->is_finished = true;

  // mark that it is done
  gpu_memory.mark_transfer_done(t1);
  gpu_memory.mark_transfer_done(t2);

  // signal the kernel prep as done
  gpu_memory.finish_kernel_prep(prep1);

  // we should not be able to preallocate
  auto dev2 = gpu_memory.can_preallocate(prep2, 0);
  EXPECT_EQ(dev2, -1);

  // check if we can garbage collect
  dev2 = gpu_memory.can_gc(prep2, 0);
  EXPECT_EQ(dev2, 0);

  // get the garbage collection request
  auto req = gpu_memory.get_gc_request(prep2, dev2);
  EXPECT_EQ(req->to_free.size(), 2);
  EXPECT_EQ(req->to_evict.size(), 1);

  // process the gc request
  fake_process_gc_request(req);

  // finish the garbage collection request
  gpu_memory.finish_gc_request(req);

  // we should be able to preallocate
  dev2 = gpu_memory.can_preallocate(prep2, 0);
  EXPECT_EQ(dev2, 0);

  // preallocate the krenel prep
  gpu_memory.preallocate(prep2, dev2);

  // check if we have the right number of transfers
  EXPECT_EQ(prep1->cpu_transfers.size(), 2);
  EXPECT_EQ(prep1->gpu_transfers.size(), 0);

  // get the transfers
  t1 = prep2->cpu_transfers[0];
  t2 = prep2->cpu_transfers[1];

  // make sure the transfers are properly initalized
  EXPECT_TRUE(t1->dst != nullptr);
  EXPECT_EQ(t1->dst_dev, 0);
  EXPECT_GE(t1->id, 0);
  EXPECT_EQ(t1->is_finished, false);
  EXPECT_EQ(t1->num_bytes, 256);
  EXPECT_EQ(t1->tid, 3);

  EXPECT_TRUE(t2->dst != nullptr);
  EXPECT_EQ(t2->dst_dev, 0);
  EXPECT_GE(t2->id, 0);
  EXPECT_EQ(t2->is_finished, false);
  EXPECT_EQ(t2->num_bytes, 256);
  EXPECT_EQ(t2->tid, 4);

  // mark as finished
  t1->is_finished = true;
  t2->is_finished = true;

  // mark that it is done
  gpu_memory.mark_transfer_done(t1);
  gpu_memory.mark_transfer_done(t2);

  // signal the kernel prep as done
  gpu_memory.finish_kernel_prep(prep2);
}

TEST(TestGPUMemory, TestGC2) {

  gpu_memory_t gpu_memory(2, 1024);

  // we want to allocate these
  auto prep1 = make_kernel_prep(0, {0, 1}, {256, 256}, {7}, {512});
  auto prep2 = make_kernel_prep(0, {2, 3}, {256, 256}, {8}, {512});
  auto prep3 = make_kernel_prep(0, {0, 4}, {256, 256}, {9}, {512});
  auto prep4 = make_kernel_prep(0, {2, 5}, {256, 256}, {10}, {512});

  // mark the apply for use
  gpu_memory.mark_for_use(create_apply(0, {0, 1}, {256, 256}, {7}, {512}));
  gpu_memory.mark_for_use(create_apply(0, {2, 3}, {256, 256}, {8}, {512}));
  gpu_memory.mark_for_use(create_apply(0, {0, 4}, {256, 256}, {9}, {512}));
  gpu_memory.mark_for_use(create_apply(0, {2, 5}, {256, 256}, {10}, {512}));

  // mark that these tensors are loaded on the CPU
  gpu_memory.tensor_loaded_on_cpu(0, 256);
  gpu_memory.tensor_loaded_on_cpu(1, 256);
  gpu_memory.tensor_loaded_on_cpu(2, 256);
  gpu_memory.tensor_loaded_on_cpu(3, 256);
  gpu_memory.tensor_loaded_on_cpu(4, 256);
  gpu_memory.tensor_loaded_on_cpu(5, 256);

  // check if we can preallocate prep1 (we should be able to)
  auto dev1 = gpu_memory.can_preallocate(prep1, 0);
  EXPECT_NE(dev1, -1);

  // preallocate the krenel prep1
  gpu_memory.preallocate(prep1, dev1);

  // check if we can preallocate prep2 (we should be able to)
  auto dev2 = gpu_memory.can_preallocate(prep2, 0);
  EXPECT_NE(dev2, -1);

  // preallocate the krenel prep2
  gpu_memory.preallocate(prep2, dev2);

  // the devices should not be the same
  EXPECT_NE(dev1, dev2);

  // check if we have the right number of transfers
  EXPECT_EQ(prep1->cpu_transfers.size(), 2);
  EXPECT_EQ(prep1->gpu_transfers.size(), 0);
  EXPECT_EQ(prep2->cpu_transfers.size(), 2);
  EXPECT_EQ(prep2->gpu_transfers.size(), 0);

  // get the transfers
  auto t1 = prep1->cpu_transfers[0];
  auto t2 = prep1->cpu_transfers[1];
  auto t3 = prep2->cpu_transfers[0];
  auto t4 = prep2->cpu_transfers[1];

  // make sure the transfers are properly initalized
  EXPECT_TRUE(t1->dst != nullptr);
  EXPECT_EQ(t1->dst_dev, dev1);
  EXPECT_GE(t1->id, 0);
  EXPECT_EQ(t1->is_finished, false);
  EXPECT_EQ(t1->num_bytes, 256);
  EXPECT_EQ(t1->tid, 0);

  EXPECT_TRUE(t2->dst != nullptr);
  EXPECT_EQ(t2->dst_dev, dev1);
  EXPECT_GE(t2->id, 0);
  EXPECT_EQ(t2->is_finished, false);
  EXPECT_EQ(t2->num_bytes, 256);
  EXPECT_EQ(t2->tid, 1);

  EXPECT_TRUE(t3->dst != nullptr);
  EXPECT_EQ(t3->dst_dev, dev2);
  EXPECT_GE(t3->id, 0);
  EXPECT_EQ(t3->is_finished, false);
  EXPECT_EQ(t3->num_bytes, 256);
  EXPECT_EQ(t3->tid, 2);

  EXPECT_TRUE(t4->dst != nullptr);
  EXPECT_EQ(t4->dst_dev, dev2);
  EXPECT_GE(t4->id, 0);
  EXPECT_EQ(t4->is_finished, false);
  EXPECT_EQ(t4->num_bytes, 256);
  EXPECT_EQ(t4->tid, 3);

  // mark as finished
  t1->is_finished = true;
  t2->is_finished = true;
  t3->is_finished = true;
  t4->is_finished = true;

  // mark that it is done
  gpu_memory.mark_transfer_done(t1);
  gpu_memory.mark_transfer_done(t2);
  gpu_memory.mark_transfer_done(t3);
  gpu_memory.mark_transfer_done(t4);

  // finish the kernel prep
  gpu_memory.finish_kernel_prep(prep1);
  gpu_memory.finish_kernel_prep(prep2);

  // check if we can garbage collect
  auto dev3 = gpu_memory.can_gc(prep3, 0);
  EXPECT_EQ(dev3, 0);

  // get the garbage collection request
  auto req1 = gpu_memory.get_gc_request(prep3, dev3);
  EXPECT_EQ(req1->to_free.size(), 1);
  EXPECT_EQ(req1->to_evict.size(), 1);

  // check if we can garbage collect
  auto dev4 = gpu_memory.can_gc(prep4, 0);
  EXPECT_EQ(dev3, 0);

  // get the garbage collection request
  auto req2 = gpu_memory.get_gc_request(prep4, dev4);
  EXPECT_EQ(req2->to_free.size(), 1);
  EXPECT_EQ(req2->to_evict.size(), 1);

  // finish the garbage collection request and immediately preallocate
  gpu_memory.finish_gc_request(req1);
  gpu_memory.preallocate(req1->to_run, req1->dev);

  // finish the garbage collection request and immediately preallocate
  gpu_memory.finish_gc_request(req2);
  gpu_memory.preallocate(req2->to_run, req2->dev);

  // get the transfers
  auto t5 = req1->to_run->cpu_transfers[0];
  auto t6 = req2->to_run->cpu_transfers[0];

  // mark as finished
  t5->is_finished = true;
  t6->is_finished = true;

  // mark that it is done
  gpu_memory.mark_transfer_done(t5);
  gpu_memory.mark_transfer_done(t6);

  // finish the kernel prep
  gpu_memory.finish_kernel_prep(req1->to_run);
  gpu_memory.finish_kernel_prep(req2->to_run);
}