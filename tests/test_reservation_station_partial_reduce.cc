#include "../src/commands/reservation_station.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <unordered_map>

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
  rs->queue_command(std::move(reduce));
  rs->execute_scheduled_async();

  // get the next kernel
  auto k1 = rs->get_next_kernel_command();

  // fake run the kernel
  k1->get_output(0).tid = -5;
  values[k1->get_output(0).tid] = values[k1->get_input(0).tid] + values[k1->get_input(1).tid];

  // retire the kernel
  rs->retire_command(std::move(k1));

  // register the tensor
  rs->register_tensor(3);

  // get the next kernel
  auto k2 = rs->get_next_kernel_command();
  values[k2->get_output(0).tid] = values[k2->get_input(0).tid] + values[k2->get_input(1).tid];

  // retire the command
  rs->retire_command(std::move(k2));

  // check if everything went fine
  EXPECT_EQ(values[4], 16);
}

TEST(TestReservationStation, TestTwoNodes) {
  
  auto reduce =
      command_t::create_reduce(0, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                command_t::tid_node_id_t{.tid = 2, .node = 0},
                                command_t::tid_node_id_t{.tid = 3, .node = 0},
                                command_t::tid_node_id_t{.tid = 4, .node = 1},
                                command_t::tid_node_id_t{.tid = 5, .node = 1}},
                               {command_t::tid_node_id_t{.tid = 4, .node = 0}});
}