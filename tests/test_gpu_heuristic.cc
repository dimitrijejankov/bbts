#include <array>
#include <gtest/gtest.h>
#include "../src/gpu/gpu_heuristic.h"
#include "../src/commands/reservation_station.h"

using namespace bbts;

// just a small function to quickly create an apply
gpu_command_schedule_ptr_t create_apply(command_id_t id,
                                  const std::vector<tid_t> &inputs, 
                                  const std::vector<tid_t> &outputs) {

  std::vector<command_t::tid_node_id_t> in;
  for(auto &i : inputs) {
    in.push_back(command_t::tid_node_id_t{.tid = i, .node = 0});
  }

  std::vector<command_t::tid_node_id_t> out;
  for(auto &o : outputs) {
    out.push_back(command_t::tid_node_id_t{.tid = o, .node = 0});
  }

  // make the actual apply schedule
  auto apply = std::make_shared<bbts::gpu_command_schedule_t>(); 
  apply->cmd = command_t::create_apply(id, {0, 0}, true, {}, in, out);

  // we don't care about this
  apply->input_sizes.resize(inputs.size());
  apply->output_sizes.resize(outputs.size());
  apply->fn = nullptr;

  return std::move(apply);
}

gpu_command_schedule_ptr_t create_reduce(command_id_t id,
                                  const std::vector<tid_t> &inputs, 
                                  tid_t output) {

  std::vector<command_t::tid_node_id_t> in;
  for(auto &i : inputs) {
    in.push_back(command_t::tid_node_id_t{.tid = i, .node = 0});
  }

  // make the actual reduce schedule
  auto reduce = std::make_shared<bbts::gpu_command_schedule_t>(); 
  reduce->cmd = command_t::create_reduce(id, {0, 0}, true, {}, in, command_t::tid_node_id_t{.tid = output, .node = 0});

  // we don't care about this
  reduce->input_sizes.resize(inputs.size());
  reduce->output_sizes.resize(1);
  reduce->params = {};
  reduce->fn = nullptr;

  return std::move(reduce);
}

TEST(TestGPUHeuristic, TestPerGPU) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_apply(id++, {0, 1}, {8});
  heuristic.register_apply(cmd1);

  auto cmd2 = create_apply(id++, {2, 3}, {9});
  heuristic.register_apply(cmd2);

  auto cmd3 = create_apply(id++, {4, 5}, {10});
  heuristic.register_apply(cmd3);

  auto cmd4 = create_apply(id++, {6, 7}, {11});
  heuristic.register_apply(cmd4);

  heuristic.tensor_loaded(0, 0);
  heuristic.tensor_loaded(1, 0);

  heuristic.tensor_loaded(2, 1);
  heuristic.tensor_loaded(3, 1);

  heuristic.tensor_loaded(4, 2);
  heuristic.tensor_loaded(5, 2);

  heuristic.tensor_loaded(6, 3);
  heuristic.tensor_loaded(7, 3);

  auto [k2, dev2] = heuristic.get_next_on_same(2);
  heuristic.mark_as_scheduled(k2);

  // check everything
  EXPECT_EQ(dev2, 2);

  auto [k3, dev3] = heuristic.get_next_on_same(3);
  heuristic.mark_as_scheduled(k3);

  EXPECT_EQ(dev3, 3);

  auto [k0, dev0] = heuristic.get_next_on_same(0);
  heuristic.mark_as_scheduled(k0);

  EXPECT_EQ(dev0, 0);

  auto [k1, dev1] = heuristic.get_next_on_same(1);
  heuristic.mark_as_scheduled(k1);

  EXPECT_EQ(dev1, 1);

  auto [k_none, dev_none] = heuristic.get_next_on_same(0);
  EXPECT_EQ(dev_none, -1);
}

TEST(TestGPUHeuristic, TestBetweenGPU) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_apply(id++, {0, 1}, {8});
  heuristic.register_apply(cmd1);

  auto cmd2 = create_apply(id++, {2, 3}, {9});
  heuristic.register_apply(cmd2);

  auto cmd3 = create_apply(id++, {4, 5}, {10});
  heuristic.register_apply(cmd3);

  auto cmd4 = create_apply(id++, {6, 7}, {11});
  heuristic.register_apply(cmd4);

  heuristic.tensor_loaded(0, 0);
  heuristic.tensor_loaded(1, 1);

  heuristic.tensor_loaded(2, 0);
  heuristic.tensor_loaded(3, 1);

  heuristic.tensor_loaded(4, 3);
  heuristic.tensor_loaded(5, 2);

  heuristic.tensor_loaded(6, 3);
  heuristic.tensor_loaded(7, 2);

  auto [k2, dev2] = heuristic.get_next_on_same(2);

  // check everything
  EXPECT_EQ(k2, nullptr);
  EXPECT_EQ(dev2, -1);

  auto [k3, dev3] = heuristic.get_next_on_same(3);

  // check everything
  EXPECT_EQ(k3, nullptr);
  EXPECT_EQ(dev3, -1);

  auto [k0, dev0] = heuristic.get_next_on_same(0);

  // check everything
  EXPECT_EQ(k0, nullptr);
  EXPECT_EQ(dev0, -1);

  auto [k1, dev1] = heuristic.get_next_on_same(1);

  // check everything
  EXPECT_EQ(k1, nullptr);
  EXPECT_EQ(dev1, -1);

  std::array<kernel_prep_ptr_t, 4> preps;

  // check the results
  auto k_tmp = heuristic.get_next_on_any();
  preps[k_tmp->command_id] = k_tmp;
  heuristic.mark_as_scheduled(k_tmp);

  k_tmp = heuristic.get_next_on_any();
  preps[k_tmp->command_id] = k_tmp;
  heuristic.mark_as_scheduled(k_tmp);

  k_tmp = heuristic.get_next_on_any();
  preps[k_tmp->command_id] = k_tmp;
  heuristic.mark_as_scheduled(k_tmp);

  k_tmp = heuristic.get_next_on_any();
  preps[k_tmp->command_id] = k_tmp;
  heuristic.mark_as_scheduled(k_tmp);

  // make sure they are all there
  EXPECT_EQ(preps[0]->command_id, 0);
  EXPECT_EQ(preps[1]->command_id, 1);
  EXPECT_EQ(preps[2]->command_id, 2);
  EXPECT_EQ(preps[3]->command_id, 3);
  
  auto [k_none, dev_none] = heuristic.get_next_on_same(0);
  EXPECT_EQ(dev_none, -1);
}

TEST(TestGPUHeuristic, TestMixed) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_apply(id++, {0, 1}, {8});
  heuristic.register_apply(cmd1);

  auto cmd2 = create_apply(id++, {2, 3}, {9});
  heuristic.register_apply(cmd2);

  auto cmd3 = create_apply(id++, {4, 5}, {10});
  heuristic.register_apply(cmd3);

  auto cmd4 = create_apply(id++, {6, 7}, {11});
  heuristic.register_apply(cmd4);

  heuristic.tensor_loaded(0, 0);
  heuristic.tensor_loaded(1, 0);

  heuristic.tensor_loaded(2, 0);
  heuristic.tensor_loaded(3, 1);

  heuristic.tensor_loaded(4, 3);
  heuristic.tensor_loaded(5, 2);

  heuristic.tensor_loaded(6, 3);
  heuristic.tensor_loaded(7, 3);
  
  auto [k1, dev1] = heuristic.get_next_on_same(3);
  heuristic.mark_as_scheduled(k1);

  auto [k2, dev2] = heuristic.get_next_on_same(3);
  heuristic.mark_as_scheduled(k2);

  auto [k_none, dev_none] = heuristic.get_next_on_same(3);
  EXPECT_EQ(dev_none, -1);

  auto k3 = heuristic.get_next_on_any();
  heuristic.mark_as_scheduled(k3);

  auto k4 = heuristic.get_next_on_any();
  heuristic.mark_as_scheduled(k4);

  // make sure we don't have any
  k_none = heuristic.get_next_on_any();
  EXPECT_EQ(k_none, nullptr);

  std::array<kernel_prep_ptr_t, 4> preps;
  preps[k1->command_id] = k1;
  preps[k2->command_id] = k2;
  preps[k3->command_id] = k3;
  preps[k4->command_id] = k4;

  // make sure they are all there
  EXPECT_EQ(preps[0]->command_id, 0);
  EXPECT_EQ(preps[1]->command_id, 1);
  EXPECT_EQ(preps[2]->command_id, 2);
  EXPECT_EQ(preps[3]->command_id, 3);
}


TEST(TestGPUHeuristic, TestLoadingUnloading) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_apply(id++, {0, 1}, {6});
  heuristic.register_apply(cmd1);

  auto cmd2 = create_apply(id++, {0, 2}, {7});
  heuristic.register_apply(cmd2);

  auto cmd3 = create_apply(id++, {3, 2}, {8});
  heuristic.register_apply(cmd3);

  auto cmd4 = create_apply(id++, {3, 4}, {9});
  heuristic.register_apply(cmd4);

  // 
  heuristic.tensor_loaded(0, 0);
  heuristic.tensor_loaded(1, 0);
  heuristic.tensor_loaded(2, 3);
  heuristic.tensor_loaded(3, 3);
  
  heuristic.tensor_unloaded(0, 0);
  heuristic.tensor_unloaded(3, 3);

  // make sure we have nothing
  {
    auto [k_none, dev_none] = heuristic.get_next_on_same(0);
    EXPECT_EQ(k_none, nullptr);
    EXPECT_EQ(dev_none, -1);

    k_none = heuristic.get_next_on_any();
    EXPECT_EQ(k_none, nullptr);
  }
  heuristic.tensor_unloaded(2, 3);

  // make sure we still have nothing
  {
    auto [k_none, dev_none] = heuristic.get_next_on_same(0);
    EXPECT_EQ(k_none, nullptr);
    EXPECT_EQ(dev_none, -1);

    k_none = heuristic.get_next_on_any();
    EXPECT_EQ(k_none, nullptr);
  }

  heuristic.tensor_loaded(0, 2);

  // 
  {
    auto [k_none, dev_none] = heuristic.get_next_on_same(0);
    EXPECT_EQ(k_none, nullptr);
    EXPECT_EQ(dev_none, -1);

    auto k_1 = heuristic.get_next_on_any();
    EXPECT_EQ(k_1->command_id, 0);
    heuristic.mark_as_scheduled(k_1);
  }

  heuristic.tensor_loaded(2, 2);

  // 
  {
    auto [k_2, dev_2] = heuristic.get_next_on_same(0);
    heuristic.mark_as_scheduled(k_2);
    EXPECT_EQ(k_2->command_id, 1);
    EXPECT_EQ(dev_2, 2);

    auto k_none = heuristic.get_next_on_any();
    EXPECT_EQ(k_none, nullptr);
  }

  heuristic.tensor_loaded(3, 3);

  // 
  {
    auto [k_none, dev_none] = heuristic.get_next_on_same(0);
    EXPECT_EQ(k_none, nullptr);
    EXPECT_EQ(dev_none, -1);

    auto k_3 = heuristic.get_next_on_any();
    EXPECT_EQ(k_3->command_id, 2);
    heuristic.mark_as_scheduled(k_3);
  }

  heuristic.tensor_loaded(4, 3);

  // 
  {
    auto [k_3, dev_3] = heuristic.get_next_on_same(0);
    heuristic.mark_as_scheduled(k_3);
    EXPECT_EQ(k_3->command_id, 3);
    EXPECT_EQ(dev_3, 3);

    auto k_none = heuristic.get_next_on_any();
    EXPECT_EQ(k_none, nullptr);
  }

}

TEST(TestGPUHeuristic, TestReduce1) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_reduce(id++, {0, 1, 2, 3}, 4);
  heuristic.register_reduce(cmd1);

  heuristic.tensor_loaded(0, 0);
  heuristic.tensor_loaded(1, 1);
  heuristic.tensor_loaded(2, 2);
  heuristic.tensor_loaded(3, 3);

  // tid: 0 + tid: 1 -> tid: -1
  auto k_1 = heuristic.get_next_on_any();
  heuristic.mark_as_scheduled(k_1);

  // tid: 2 + tid: 3 -> tid: -2
  auto k_2 = heuristic.get_next_on_any();
  heuristic.mark_as_scheduled(k_2);

  // make sure the outputs are anonymous tensors (less than 4)
  EXPECT_LE(k_1->output.front(), 0);
  EXPECT_LE(k_2->output.front(), 0);

  auto k_none = heuristic.get_next_on_any();
  EXPECT_EQ(k_none, nullptr);

  heuristic.tensor_loaded(k_1->output[0], 1);
  heuristic.tensor_loaded(k_2->output[0], 2);

  // tid: -1 + tid: -2 -> tid: 4
  auto k_3 = heuristic.get_next_on_any();
  heuristic.mark_as_scheduled(k_3);

  // remove the heuristics
  heuristic.remove_tensor(k_1->output[0]);
  heuristic.remove_tensor(k_2->output[0]);

  EXPECT_EQ(k_3->output.front(), 4);

  k_none = heuristic.get_next_on_any();
  EXPECT_EQ(k_none, nullptr);
}


TEST(TestGPUHeuristic, TestApplyHeuristic1) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  // load all the tensors on the CPU
  heuristic.tensor_on_cpu(1);
  heuristic.tensor_on_cpu(2);
  heuristic.tensor_on_cpu(3);
  heuristic.tensor_on_cpu(4);
  heuristic.tensor_on_cpu(5);
  heuristic.tensor_on_cpu(6);

  // needs 2 copies and 4 is used by 2 and 1 is used by 3, heuristic (2, 5)
  auto cmd1 = create_apply(id++, {4, 1}, {7}); 
  heuristic.register_apply(cmd1);

  // needs 2 copies and 2 is used by 1 and 1 is used by 3, heuristic is (2, 4)
  auto cmd2 = create_apply(id++, {2, 1}, {9});
  heuristic.register_apply(cmd2);

  // needs 3 copes and 3 is used by 1, 1 is used by 3 and 6 is used by 1, heuristic is (3, 5)
  // correction:  needs 3 copes and 3 is used by 1, 1 is used by 3 and 6 is used by 1, heuristic is (3, 6)
  auto cmd3 = create_apply(id++, {3, 1, 6}, {10});
  heuristic.register_apply(cmd3);

  // needs 2 copies and 4 is used by 2 and 5 used by 1, heuristic is (2, 3)
  // correction: needs 3 copies: 4 is used by 2; 5 is used by 1; 6 is used by 2; heursitic is (3, 5)
  auto cmd4 = create_apply(id++, {4, 5, 6}, {11});
  heuristic.register_apply(cmd4);

  auto [k_none_1, dev_none_1] = heuristic.get_next_on_same(0);
  EXPECT_EQ(dev_none_1, -1);

  // we expect this to be APPLY (4, 1) -> 7
  auto k1 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k1);
  EXPECT_EQ(k1->command_id, 0);

  // load the tensors
  heuristic.tensor_loaded(4, 0);
  heuristic.tensor_loaded(1, 0);

  // we expect this to be APPLY (2, 1) -> 9
  auto k2 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k2);
  EXPECT_EQ(k2->command_id, 1);

  // load the tensors
  heuristic.tensor_loaded(1, 0);

  // we expect this to be APPLY (3, 1, 6) -> 10
  auto k3 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k3);
  EXPECT_EQ(k3->command_id, 3);

  // load the tensors
  heuristic.tensor_loaded(3, 0);
  heuristic.tensor_loaded(6, 0);

  // we expect this to be APPLY (4, 5, 6) -> 11
  auto k4 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k4);
  EXPECT_EQ(k4->command_id, 2);

  auto [k_none_2, dev_none_2] = heuristic.get_next_on_same(0);
  EXPECT_EQ(dev_none_2, -1);
}

TEST(TestGPUHeuristic, TestApplyHeuristic2) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  auto cmd1 = create_apply(id++, {4, 1}, {7});
  heuristic.register_apply(cmd1);

  auto cmd2 = create_apply(id++, {2, 1}, {8});
  heuristic.register_apply(cmd2);

  auto cmd3 = create_apply(id++, {3, 1, 6}, {9});
  heuristic.register_apply(cmd3);

  auto cmd4 = create_apply(id++, {4, 5}, {10});
  heuristic.register_apply(cmd4);

  auto [k_none_1, dev_none_1] = heuristic.get_next_on_same(0);
  EXPECT_EQ(dev_none_1, -1);

  k_none_1 = heuristic.get_next_heuristic();
  EXPECT_EQ(k_none_1, nullptr);

  // load all the tensors on the CPU
  heuristic.tensor_on_cpu(1);
  heuristic.tensor_on_cpu(2);
  heuristic.tensor_on_cpu(3);
  heuristic.tensor_on_cpu(4);
  heuristic.tensor_on_cpu(5);
  heuristic.tensor_on_cpu(6);

  auto k1 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k1);
  EXPECT_EQ(k1->command_id, 0);

  auto k2 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k2);
  EXPECT_EQ(k2->command_id, 1);

  auto k3 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k3);
  EXPECT_EQ(k3->command_id, 3);

  auto k4 = heuristic.get_next_heuristic();
  heuristic.mark_as_scheduled(k4);
  EXPECT_EQ(k4->command_id, 2);

  auto k_none_2 = heuristic.get_next_heuristic();
  EXPECT_EQ(k_none_2, nullptr);
}

TEST(TestGPUHeuristic, TestApplyHeuristic3) {

  command_id_t id = 0;
  bbts::gpu_heuristic_t heuristic(4);

  // load all the tensors on the CPU
  heuristic.tensor_on_cpu(1);
  heuristic.tensor_on_cpu(2);

  // register the reduce
  auto cmd1 = create_reduce(id++, {1, 2, 3, 4}, 5);
  heuristic.register_reduce(cmd1);

  // tid: 1 + tid: 2 -> tid: -1
  auto k_1 = heuristic.get_next_heuristic();

  // load the tensors and retire
  heuristic.tensor_loaded(k_1->input[0], 0);
  heuristic.tensor_loaded(k_1->input[1], 0);
  heuristic.mark_as_scheduled(k_1);

  // load the output tensor
  heuristic.tensor_loaded(k_1->output[0], 0);
  
  // tid: -1 + tid: 3 -> tid: -2
  heuristic.tensor_on_cpu(3);
  auto k_2 = heuristic.get_next_heuristic();

  // load the next tensor and mark as scheduled
  heuristic.tensor_loaded(k_2->input[0] < 0 ? k_2->input[1] : k_2->input[0], 0);
  heuristic.mark_as_scheduled(k_2);

  // load the output tensor
  heuristic.tensor_loaded(k_2->output[0], 0);

  // tid: -2 + tid: 4 -> tid: 5
  heuristic.tensor_on_cpu(4);
  auto k_3 = heuristic.get_next_heuristic();

  // load the next tensor and mark as scheduled
  heuristic.tensor_loaded(k_3->input[0] < 0 ? k_3->input[1] : k_3->input[0], 0);
  heuristic.mark_as_scheduled(k_3);

  auto k_none = heuristic.get_next_heuristic();
  EXPECT_EQ(k_none, nullptr);
}