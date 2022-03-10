#include <array>
#include <gtest/gtest.h>
#include "../src/gpu/gpu_heuristic.h"
#include "../src/commands/reservation_station.h"

using namespace bbts;

// just a small function to quickly create an apply
apply_schedule_ptr_t create_apply(command_id_t id,
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
  auto apply = std::make_shared<bbts::apply_schedule_t>(); 
  apply->cmd = command_t::create_apply(id, {0, 0}, true, {}, in, out);

  // we don't care about this
  apply->input_num_bytes = {};
  apply->output_num_bytes = {};
  apply->fn = nullptr;

  return std::move(apply);
}

apply_schedule_ptr_t create_reduce(command_id_t id,
                                  const std::vector<tid_t> &inputs, 
                                  tid_t output) {

  std::vector<command_t::tid_node_id_t> in;
  for(auto &i : inputs) {
    in.push_back(command_t::tid_node_id_t{.tid = i, .node = 0});
  }

  // make the actual apply schedule
  auto reduce = std::make_shared<bbts::apply_schedule_t>(); 
  reduce->cmd = command_t::create_reduce(id, {0, 0}, true, {}, in, command_t::tid_node_id_t{.tid = output, .node = 0});

  // we don't care about this
  reduce->input_num_bytes = {};
  reduce->output_num_bytes = {};
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