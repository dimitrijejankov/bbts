#include "../main/commands/reservation_station.h"
#include <cstddef>
#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace bbts;

TEST(TestReorderBuffer, TestOne) {

    auto apply_1 = command_t::create_apply(0, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 9, .node = 0},
                                            command_t::tid_node_id_t{.tid = 10, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 1, .node = 0}});

    auto apply_2 = command_t::create_apply(1, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 11, .node = 0},
                                            command_t::tid_node_id_t{.tid = 12, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 2, .node = 0}});

    auto apply_3 = command_t::create_apply(2, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 13, .node = 0},
                                            command_t::tid_node_id_t{.tid = 14, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 3, .node = 0}});

    auto apply_4 = command_t::create_apply(3, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 15, .node = 0},
                                            command_t::tid_node_id_t{.tid = 16, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 4, .node = 0}});

    auto apply_5 = command_t::create_apply(4, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 16, .node = 0},
                                            command_t::tid_node_id_t{.tid = 17, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 5, .node = 0}});

    auto apply_6 = command_t::create_apply(5, {-1, -1}, false, {}, 
                                           {command_t::tid_node_id_t{.tid = 18, .node = 0},
                                            command_t::tid_node_id_t{.tid = 19, .node = 0}}, 
                                           {command_t::tid_node_id_t{.tid = 6, .node = 0}});

    reorder_buffer_t reorder_buffer;
    auto reduce_1 =
      command_t::create_reduce(8, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 1, .node = 0},
                                command_t::tid_node_id_t{.tid = 2, .node = 0},
                                command_t::tid_node_id_t{.tid = 3, .node = 0}},
                               {command_t::tid_node_id_t{.tid = 7, .node = 0}});

    auto reduce_2 =
      command_t::create_reduce(9, {0, 0}, false, {},
                               {command_t::tid_node_id_t{.tid = 4, .node = 0},
                                command_t::tid_node_id_t{.tid = 5, .node = 0},
                                command_t::tid_node_id_t{.tid = 6, .node = 0}},
                               {command_t::tid_node_id_t{.tid = 8, .node = 0}});


    std::vector<command_ptr_t> commands;
    commands.push_back(apply_1->clone());
    commands.push_back(apply_2->clone());
    commands.push_back(apply_3->clone());
    commands.push_back(apply_4->clone());
    commands.push_back(apply_5->clone());
    commands.push_back(apply_6->clone());
    commands.push_back(reduce_1->clone());
    commands.push_back(reduce_2->clone());

    reorder_buffer.analyze(commands);
    reorder_buffer.execute();

    reorder_buffer.queue(apply_1->clone());

    command_ptr_t out;
    reorder_buffer.get_next(command_t::op_type_t::APPLY, out);

    EXPECT_EQ(out->id, apply_1->id);

    reorder_buffer.queue(apply_4->clone());
    reorder_buffer.queue(apply_5->clone());
    reorder_buffer.queue(apply_6->clone());
    reorder_buffer.queue(apply_2->clone());
    
    reorder_buffer.get_next(command_t::op_type_t::APPLY, out);
    EXPECT_EQ(out->id, apply_2->id);

    reorder_buffer.get_next(command_t::op_type_t::APPLY, out);
    EXPECT_EQ(out->id, apply_4->id);

    reorder_buffer.get_next(command_t::op_type_t::APPLY, out);
    EXPECT_EQ(out->id, apply_5->id);

    reorder_buffer.get_next(command_t::op_type_t::APPLY, out);
    EXPECT_EQ(out->id, apply_6->id);
}