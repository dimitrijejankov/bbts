#include "../src/commands/reservation_station.h"
#include <gtest/gtest.h>

using namespace bbts;

TEST(TestReservationStation, TwoNodesCMM) {

  auto reduce =
      command_t::create_reduce(0, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                command_t::tid_node_id_t{.tid = 2, .node = 0},
                                command_t::tid_node_id_t{.tid = 3, .node = 0}},
                               {command_t::tid_node_id_t{.tid = 4, .node = 0}});

  // make a reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // register the tensor
  rs->register_tensor(0);
  rs->register_tensor(1);
  rs->register_tensor(2);

  rs->queue_command(std::move(reduce));
}