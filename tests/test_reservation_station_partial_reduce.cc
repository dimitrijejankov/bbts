#include "../main/commands/reservation_station.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>

using namespace bbts;

TEST(TestReservationStation, TestSingleNode) {

  std::unordered_map<tid_t, size_t> values = {{1, 4}, {2, 5}, {3, 7}};
  auto reduce =
      command_t::create_reduce(0, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                command_t::tid_node_id_t{.tid = 2, .node = 0},
                                command_t::tid_node_id_t{.tid = 3, .node = 0}},
                               {command_t::tid_node_id_t{.tid = 4, .node = 0}});

  // make a reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // register the tensor
  rs->register_tensor(1);
  rs->register_tensor(2);

  // add the reduce command
  std::vector<command_ptr_t> tmp; tmp.push_back(std::move(reduce));
  rs->queue_commands(tmp);
  rs->execute_scheduled_async();

  // get the next kernel
  auto k1 = rs->get_next_command(command_t::op_type_t::APPLY);

  // fake run the kernel
  k1->get_output(0).tid = -5;
  values[k1->get_output(0).tid] = values[k1->get_input(0).tid] + values[k1->get_input(1).tid];

  // retire the kernel
  rs->retire_command(std::move(k1));

  // register the tensor
  rs->register_tensor(3);

  // get the next kernel
  auto k2 = rs->get_next_command(command_t::op_type_t::APPLY);
  values[k2->get_output(0).tid] = values[k2->get_input(0).tid] + values[k2->get_input(1).tid];

  // retire the command
  rs->retire_command(std::move(k2));

  // check if everything went fine
  EXPECT_EQ(values[4], 16);
}

TEST(TestReservationStation, TestTwoNodes) {
  
  std::unordered_map<tid_t, size_t> values0 = {{1, 4}, {2, 5}, {3, 7}};
  std::unordered_map<tid_t, size_t> values1 = {{4, 12}, {5, 16}};
  auto reduce =
      command_t::create_reduce(0, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                command_t::tid_node_id_t{.tid = 2, .node = 0},
                                command_t::tid_node_id_t{.tid = 3, .node = 0},
                                command_t::tid_node_id_t{.tid = 4, .node = 1},
                                command_t::tid_node_id_t{.tid = 5, .node = 1}},
                               {command_t::tid_node_id_t{.tid = 6, .node = 0}});

  // make a reservation station
  auto rs0 = std::make_shared<reservation_station_t>(0, 2);
  auto rs1 = std::make_shared<reservation_station_t>(1, 2);

  // register the tensor
  rs0->register_tensor(1);
  rs0->register_tensor(2);
  rs1->register_tensor(4);
  rs1->register_tensor(5);

  // add the reduce command
  std::vector<command_ptr_t> tmp; tmp.push_back(std::move(reduce));
  rs0->queue_commands(tmp);
  rs0->execute_scheduled_async();

  // add the reduce command
  rs1->queue_commands(tmp);
  rs1->execute_scheduled_async();

  // get the next kernel
  auto k1 = rs0->get_next_command(command_t::op_type_t::APPLY);

  // fake run the kernel
  k1->get_output(0).tid = -5;
  values0[k1->get_output(0).tid] = values0[k1->get_input(0).tid] + values0[k1->get_input(1).tid];
  rs0->retire_command(std::move(k1));

  // get the next kernel
  auto k2 = rs1->get_next_command(command_t::op_type_t::APPLY);

  // fake run it
  k2->get_output(0).tid = -5;
  values1[k2->get_output(0).tid] = values1[k2->get_input(0).tid] + values1[k2->get_input(1).tid];
  rs1->retire_command(std::move(k2));

  // register the tensor
  rs0->register_tensor(3);

  // get the next kernel
  auto k3 = rs0->get_next_command(command_t::op_type_t::APPLY);

  // fake run it
  k3->get_output(0).tid = -6;
  values0[k3->get_output(0).tid] = values0[k3->get_input(0).tid] + values0[k3->get_input(1).tid];
  rs0->retire_command(std::move(k3));

  // get the notifications
  bool is_done;
  auto notifications = rs1->commands_ready_for_node(0, is_done);

  // notify that reduce is done
  rs0->notify_ready_command(1, notifications);
  auto k4_0 = rs0->get_next_command(command_t::op_type_t::REDUCE);

  // clone this one so we can retire it...
  auto k4_1 = k4_0->clone();

  auto in1 = k4_0->get_input(0);
  auto in2 = k4_0->get_input(1);

  // retire the commands
  rs0->retire_command(std::move(k4_0));
  rs1->retire_command(std::move(k4_1));
}