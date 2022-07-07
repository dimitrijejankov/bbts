#include <gtest/gtest.h>
#include "../src/commands/parsed_command.h"

namespace bbts {

TEST(TestCommandParsing, Test1) {

  {
    parsed_command_list_t cmd_list;

    cmd_list.add_move({0,3}, {{0,0}, {0, 1}, {0, 2}});
    cmd_list.add_apply("bla",
                       {"dense", "sparse"},
                       {"sparse", "dense", "dense"},
                       false,
                       {{0,0}, {0, 1}},
                       {{0,2}, {0, 3}, {0, 4}},
                       {command_param_t{.u = 1}, command_param_t{.i = 2}, command_param_t{.f = 3.0f}});

    cmd_list.add_delete({{1,0}, {1, 1}, {1, 2}});
    cmd_list.add_reduce("bla2",
                        {"dense1", "sparse1"},
                        {"sparse1", "dense2"},
                        false,
                        {{0,0}, {0, 1}},
                        {0, 3},
                        {});

    cmd_list.serialize("out.bbts");
  }

  {
    parsed_command_list_t cmd_list;
    cmd_list.deserialize("out.bbts");

    /// 1. check the first command

    EXPECT_EQ(cmd_list[0].type, parsed_command_t::op_type_t::MOVE);
    EXPECT_EQ(cmd_list[0].def.ud_name, "");
    EXPECT_EQ(cmd_list[0].def.output_types.size(), 0);
    EXPECT_EQ(cmd_list[0].def.input_types.size(), 0);
    EXPECT_EQ(cmd_list[0].inputs.size(), 1);

    EXPECT_EQ(std::get<0>(cmd_list[0].inputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[0].inputs[0]), 3);

    EXPECT_EQ(std::get<0>(cmd_list[0].outputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[0].outputs[0]), 0);

    EXPECT_EQ(std::get<0>(cmd_list[0].outputs[1]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[0].outputs[1]), 1);

    EXPECT_EQ(std::get<0>(cmd_list[0].outputs[2]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[0].outputs[2]), 2);

    EXPECT_EQ(cmd_list[0].parameters.size(), 0);

    /// 2. check the second command

    EXPECT_EQ(cmd_list[1].type, parsed_command_t::op_type_t::APPLY);
    EXPECT_EQ(cmd_list[1].def.ud_name, "bla");

    EXPECT_EQ(cmd_list[1].def.input_types.size(), 2);
    EXPECT_EQ(cmd_list[1].def.input_types[0], "dense");
    EXPECT_EQ(cmd_list[1].def.input_types[1], "sparse");

    EXPECT_EQ(cmd_list[1].def.output_types.size(), 3);
    EXPECT_EQ(cmd_list[1].def.output_types[0], "sparse");
    EXPECT_EQ(cmd_list[1].def.output_types[1], "dense");
    EXPECT_EQ(cmd_list[1].def.output_types[2], "dense");

    EXPECT_EQ(cmd_list[1].inputs.size(), 2);

    EXPECT_EQ(std::get<0>(cmd_list[1].inputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[1].inputs[0]), 0);

    EXPECT_EQ(std::get<0>(cmd_list[1].inputs[1]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[1].inputs[1]), 1);

    EXPECT_EQ(cmd_list[1].outputs.size(), 3);

    EXPECT_EQ(std::get<0>(cmd_list[1].outputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[1].outputs[0]), 2);

    EXPECT_EQ(std::get<0>(cmd_list[1].outputs[1]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[1].outputs[1]), 3);

    EXPECT_EQ(std::get<0>(cmd_list[1].outputs[2]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[1].outputs[2]), 4);

    EXPECT_EQ(cmd_list[1].parameters.size(), 3);

    EXPECT_EQ(cmd_list[1].parameters[0].u, 1);
    EXPECT_EQ(cmd_list[1].parameters[1].i, 2);
    EXPECT_EQ(cmd_list[1].parameters[2].f, 3.0f);

    /// 3. check the third command

    EXPECT_EQ(cmd_list[2].type, parsed_command_t::op_type_t::DELETE);
    EXPECT_EQ(cmd_list[2].def.ud_name, "");
    EXPECT_EQ(cmd_list[2].def.output_types.size(), 0);
    EXPECT_EQ(cmd_list[2].def.input_types.size(), 0);
    EXPECT_EQ(cmd_list[2].inputs.size(), 3);

    EXPECT_EQ(std::get<0>(cmd_list[2].inputs[0]), 1);
    EXPECT_EQ(std::get<1>(cmd_list[2].inputs[0]), 0);

    EXPECT_EQ(std::get<0>(cmd_list[2].inputs[1]), 1);
    EXPECT_EQ(std::get<1>(cmd_list[2].inputs[1]), 1);

    EXPECT_EQ(std::get<0>(cmd_list[2].inputs[2]), 1);
    EXPECT_EQ(std::get<1>(cmd_list[2].inputs[2]), 2);

    EXPECT_EQ(cmd_list[2].parameters.size(), 0);

    /// 4. check the fourth command

    EXPECT_EQ(cmd_list[3].type, parsed_command_t::op_type_t::REDUCE);
    EXPECT_EQ(cmd_list[3].def.ud_name, "bla2");

    EXPECT_EQ(cmd_list[3].def.input_types.size(), 2);
    EXPECT_EQ(cmd_list[3].def.input_types[0], "dense1");
    EXPECT_EQ(cmd_list[3].def.input_types[1], "sparse1");

    EXPECT_EQ(cmd_list[3].def.output_types.size(), 2);
    EXPECT_EQ(cmd_list[3].def.output_types[0], "sparse1");
    EXPECT_EQ(cmd_list[3].def.output_types[1], "dense2");

    EXPECT_EQ(cmd_list[3].inputs.size(), 2);

    EXPECT_EQ(std::get<0>(cmd_list[3].inputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[3].inputs[0]), 0);

    EXPECT_EQ(std::get<0>(cmd_list[3].inputs[1]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[3].inputs[1]), 1);

    EXPECT_EQ(cmd_list[3].outputs.size(), 1);

    EXPECT_EQ(std::get<0>(cmd_list[3].outputs[0]), 0);
    EXPECT_EQ(std::get<1>(cmd_list[3].outputs[0]), 3);

    EXPECT_EQ(cmd_list[3].parameters.size(), 0);
  }

}

}