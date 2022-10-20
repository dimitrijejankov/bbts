#include <cstddef>
#include <cstdint>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iostream>
#include <vector>
#include <map>

#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"

using namespace bbts;

using matrix_index = std::map<std::tuple<int32_t, int32_t>, bbts::tid_t>;
using matrix_reduce_index = std::map<std::tuple<int32_t, int32_t>, std::vector<bbts::tid_t>>;

bbts::tid_t currentTID = 0;

const int32_t UNIFORM_RAND = 0;
const int32_t MERGE_SPLIT_BLOCKS = 1;

matrix_index generate_random(abstract_ud_spec_id_t ud,
                             std::vector<abstract_command_t> &commands,
                             size_t num_rows, size_t num_cols,
                             size_t split_rows, size_t split_cols) {

  matrix_index out;
  for (size_t rowID = 0; rowID < split_rows; ++rowID) {
    for (size_t colID = 0; colID < split_cols; ++colID) {

      // store the index
      auto tid = currentTID++;
      out[{rowID, colID}] = tid;

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = ud,
                             .type = abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {tid},
                             .params = {command_param_t{.u = (std::uint32_t)(num_rows / split_rows)},
                                        command_param_t{.u = (std::uint32_t) (num_cols / split_cols)},
                                        command_param_t{.f = 0.0f},
                                        command_param_t{.f = 1.0f}}});
    }
  }

  return std::move(out);
}

void generate_touch(abstract_ud_spec_id_t ud, size_t split_rows, size_t split_cols, 
                    size_t num_x_to_merge, size_t num_y_to_split, matrix_index &matrix,
                    std::vector<abstract_command_t> &commands) {

  for (size_t rowID = 0; rowID < split_rows; rowID += num_x_to_merge) {
    for (size_t colID = 0; colID < split_cols; colID++) {
      
      // inputs
      std::vector<tid_t> inputs;
      for(size_t x = 0; x < num_x_to_merge; ++x) {
        inputs.push_back(matrix[{rowID + x, colID}]);
      }

      // outputs
      std::vector<tid_t> outputs;
      for(size_t y = colID * num_y_to_split; y < (colID + 1) * num_y_to_split; ++y) {
          outputs.push_back(matrix[{rowID / num_x_to_merge, y}]);
      }

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = ud,
                             .type = abstract_command_type_t::TOUCH,
                             .input_tids = inputs,
                             .output_tids = outputs,
                             .params = {}});
    }
  }
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "Incorrect usage\n";
    std::cout << "Usage ./generate_test_touch <matrix_size> <split> <num_x_to_merge> <num_y_to_split>\n";
    return -1;
  }

  // parse the size and split
  char *end;
  auto matrix_size = std::strtol(argv[1], &end, 10);
  auto split = std::strtol(argv[2], &end, 10);
  auto num_x_to_merge = std::strtol(argv[3], &end, 10);
  auto num_y_to_split = std::strtol(argv[4], &end, 10);

  // 
  if(split % num_x_to_merge != 0 ||  split % num_x_to_merge  != 0 ||  matrix_size % split == 0) {
    return -1;
  }

  // the functions
  std::vector<abstract_ud_spec_t> funs;
  funs.push_back(abstract_ud_spec_t{.id = UNIFORM_RAND,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{.id = MERGE_SPLIT_BLOCKS,
                                    .ud_name = "merge_split",
                                    .input_types = {"dense"},
                                    .output_types = {"dense"}});  


  std::vector<abstract_command_t> generate_matrices;

  // generate the input batch
  auto x = generate_random(UNIFORM_RAND, generate_matrices, matrix_size,
                           matrix_size, matrix_size / split,
                           matrix_size / split);

  // write out the commands
  std::ofstream gen("gen.sbbts");
  compile_source_file_t gsf{.function_specs = funs,
                            .commands = generate_matrices};
  gsf.write_to_file(gen);
  gen.close();

  std::vector<abstract_command_t> touch_commands;
  generate_touch(MERGE_SPLIT_BLOCKS, split, split, num_x_to_merge, num_y_to_split, x, touch_commands);

  // write out the commands
  std::ofstream touch("run.sbbts");
  compile_source_file_t tsf{.function_specs = funs,
                            .commands = touch_commands};
  tsf.write_to_file(touch);
  gen.close();
}