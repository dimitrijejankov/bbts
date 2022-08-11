#include <map>
#include <sstream>
#include <thread>
#include <chrono>
#include <fstream>
#include "../../src/operations/move_op.h"
#include "../../src/operations/reduce_op.h"
#include "../../src/commands/parsed_command.h"
#include "../../src/commands/command_runner.h"

using namespace bbts;

int32_t tid_offset = 0;
using index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>>;
using to_agg_index_t = std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>>;

// creates the matrix tensors on this node
index_t create_matrix_tensors(int n,
                              int split,
                              int &cur_tid,
                              bbts::parsed_command_list_t &_cmds) {

  // the index
  index_t index;

  // block size
  uint32_t block_size = n / split;

  // create all the rows an columns we need
  auto hash_fn = std::hash<int>();
  for (int row_id = 0; row_id < split; ++row_id) {
    for (int col_id = 0; col_id < split; ++col_id) {

      // set the index
      index[{row_id, col_id}] = {0, cur_tid};

      // store the command // TODO params
      _cmds.add_apply("uniform",
                      {},
                      {"dense"},
                      false,
                      {},
                      {{cur_tid, 0}},
                      {command_param_t{.u = block_size},
                       command_param_t{.u = block_size},
                       command_param_t{.f = 0.0f},
                       command_param_t{.f = 1.0f}});

      // go to the next one
      cur_tid++;
    }
  }

  // return the index
  return std::move(index);
}

to_agg_index_t create_multiply(size_t split,
                               index_t a_mat,
                               index_t b_mat,
                               int32_t &tid_offset,
                               bbts::parsed_command_list_t &_cmds,
                               std::vector<int32_t> &to_del,
                               bool gpu_mult) {

  // create all the multiply commands
  std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>> multiplies;

  // generate the applies to do the multiplication
  for (int32_t i = 0; i < split; ++i) {
    for (int32_t j = 0; j < split; ++j) {
      for (int32_t k = 0; k < split; ++k) {

        // get the tid and the node
        auto[a_node, a_tid] = a_mat[{i, k}];
        auto[b_node, b_tid] = b_mat[{k, j}];

        // add the command
        _cmds.add_apply("matrix_mult",
                        {"dense", "dense"},
                        {"dense"},
                        gpu_mult,
                        {{a_tid, 0}, {b_tid, 0}},
                        {{tid_offset, 0}}, {});

        // mark that we need to delete this tensor later
        to_del.push_back(tid_offset);

        // do the multiplies
        multiplies[{i, j}].push_back({tid_offset, 0});

        // go to next tid
        tid_offset++;
      }
    }
  }

  return std::move(multiplies);
}

void generate_aggregation(size_t split,
                          int32_t &tid_offset,
                          to_agg_index_t &multiplies,
                          bbts::parsed_command_list_t &_cmds,
                          bool gpu_add) {

  // create the aggregate
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // all the multiplied tensors
      auto &muls = multiplies[{rowID, colID}];

      // figure out the inputs
      std::vector<std::tuple<tid_t, node_id_t>> inputs;
      for (auto &mul : muls) {
        auto &[tid, node] = mul;
        inputs.emplace_back(tid, 0);
      }

      // create the reduce command
      _cmds.add_reduce("matrix_add",
                       {"dense", "dense"},
                       {"dense"},
                       gpu_add,
                       inputs,
                       {tid_offset, 0}, {});

      tid_offset++;
    }
  }
}

void create_delete(std::vector<int32_t> &to_del,
                   bbts::parsed_command_list_t &_cmds) {

    // store the number we need to delete
    std::vector<std::tuple<tid_t, node_id_t>> _inputs;
    _inputs.reserve(to_del.size());
    for (auto t : to_del) {
        _inputs.emplace_back(t, 0);
    }

    // remove them from node
    _cmds.add_delete(_inputs);
}

std::tuple<bbts::parsed_command_list_t, index_t, index_t> generate_matrices(size_t split, size_t matrix_size) {

  bbts::parsed_command_list_t commands;

  auto a_idx = create_matrix_tensors(matrix_size, split, tid_offset, commands);
  auto b_idx = create_matrix_tensors(matrix_size, split, tid_offset, commands);

  return {std::move(commands), a_idx, b_idx};
}

bbts::parsed_command_list_t generate_commands(index_t &a_idx, index_t b_idx, bool gpu_add, bool gpu_mult, size_t split, size_t matrix_size) {

  bbts::parsed_command_list_t commands;
  // all the tensors that we need to delete
  std::vector<int32_t> to_del;

  // create the multiply commands
  auto multiplies = create_multiply(split, a_idx, b_idx, tid_offset, commands, to_del, gpu_mult);

  // generate the aggregation
  generate_aggregation(split, tid_offset, multiplies, commands, gpu_add);

  // create the delete
  create_delete(to_del, commands);

  return std::move(commands);
}

int main(int argc, char **argv) {

  if (argc != 7) {
    std::cout << "Incorrect usage\n";
    std::cout << "Usage ./generate_bmm <gpu_add> <gpu_mult> <split> <matrix_size> <gen_file>.bbts <cmd_file>.bbts\n";
    return 0;
  }

  // get the parameters
  bool gpu_add = std::string(argv[1]) == "true" ? true : false;
  bool gpu_mult = std::string(argv[2]) == "true" ? true : false;
  
  // 
  char *end;
  auto split = std::strtol(argv[3], &end, 10);
  auto matrix_size = std::strtol(argv[4], &end, 10);

  // make the generate matrix commands
  auto [matrix_gen, a_idx, b_idx] = generate_matrices(split, matrix_size);
  matrix_gen.serialize(argv[5]);

  // make the multiply commands
  auto cmds = generate_commands(a_idx, b_idx, gpu_add, gpu_mult, split, matrix_size);
  cmds.serialize(argv[6]);

  return 0;
}