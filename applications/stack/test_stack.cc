
#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"
#include <cmath>
#include <cstdint>
#include <map>

using namespace bbts;

tid_t current_tid = 0;

const int32_t UNFORM_ID = 0;
const int32_t STACK_ID = 1;

void generate_matrix(int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     std::vector<abstract_command_t> &commands,
                     std::map<std::tuple<int32_t, int32_t>, tid_t> &index) {

  std::vector<command_param_t> param_data = {
      command_param_t{.u = (std::uint32_t)(num_row / row_split)},
      command_param_t{.u = (std::uint32_t)(num_cols / col_spilt)},
      command_param_t{.f = 1.0f}, command_param_t{.f = 2.0f}};

  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = UNFORM_ID,
                             .type = abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {current_tid++},
                             .params = param_data});
    }
  }
}

void generate_stack(std::vector<abstract_command_t> &commands, std::vector<tid_t> inputs, tid_t output_id, int32_t num_row, int32_t num_cols){
    std::vector<command_param_t> param_data = {
      command_param_t{.u = (std::uint32_t)(num_row)},
      command_param_t{.u = (std::uint32_t)(num_cols)},
      command_param_t{.f = 1.0f}, command_param_t{.f = 2.0f}};
    commands.push_back(
        abstract_command_t{.ud_id = STACK_ID,
                             .type = abstract_command_type_t::STACK,
                             .input_tids = inputs,
                             .output_tids = {output_id},
                             .params = param_data});
    
}

int main() {

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // specify functions
  funs.push_back(abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{.id = STACK_ID,
                                    .ud_name = "stack",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  // commands
  std::vector<abstract_command_t> commands;

  std::map<std::tuple<int32_t, int32_t>, tid_t> a_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> b_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> c_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> d_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> e_index;
  generate_matrix(10, 10, 1, 1, commands, a_index);
  generate_matrix(10, 10, 1, 1, commands, b_index);
  generate_matrix(10, 10, 1, 1, commands, c_index);
  generate_matrix(10, 10, 1, 1, commands, d_index);
  generate_matrix(10, 10, 1, 1, commands, e_index);
  generate_stack(commands, {0, 1, 2, 3, 4}, 5, 10, 10);

  // write out the commands
  std::ofstream gen("stack_test_5Tensors.sbbts");
  compile_source_file_t gsf{.function_specs = funs, .commands = commands};
  gsf.write_to_file(gen);
  gen.close();

  return 0;
}