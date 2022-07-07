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

using index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>>;
using to_agg_index_t = std::map<std::tuple<int32_t, int32_t>, std::vector<std::tuple<tid_t, node_id_t>>>;

// creates the matrix tensors on this node
index_t create_matrix_tensors(size_t num_nodes,
                              int n,
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

      // check if this block is on this node
      auto target_node = static_cast<node_id_t>(hash_fn(row_id * split + col_id) % num_nodes);

      // set the index
      index[{row_id, col_id}] = {target_node, cur_tid};

      // store the command // TODO params
      _cmds.add_apply("uniform",
                      {},
                      {"dense"},
                      false,
                      {},
                      {{cur_tid, target_node}},
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

// create all the broadcast commands
void create_broadcast(size_t num_nodes,
                      size_t split,
                      index_t &mat_locs,
                      bbts::parsed_command_list_t &_cmds,
                      std::vector<std::vector<int32_t>> &to_del) {


  // no need to move here
  std::vector<std::tuple<tid_t, node_id_t>> outputs;

  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // set all the nodes we need to broadcast to
      outputs.clear();
      for (node_id_t n = 0; n < num_nodes; ++n) {
        if (n != node) {
          outputs.emplace_back(tid, n);
          to_del[n].push_back(tid);
        }
      }

      // create the broadcast
      _cmds.add_move({tid, node}, outputs);
    }
  }
}

// create the shuffle
template<class fun>
void create_shuffle(size_t num_nodes,
                    size_t split,
                    fun fn,
                    index_t &mat_locs,
                    bbts::parsed_command_list_t &_cmds,
                    std::vector<std::vector<int32_t>> &to_del) {


  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // no need to move here
      auto target_node = (node_id_t) fn(rowID, colID, num_nodes);
      if (node == target_node) {
        continue;
      }

      // move it
      _cmds.add_move({tid, node}, {{tid, target_node}});

      // mark that we need to delete it later
      to_del[target_node].push_back(tid);
    }
  }
}

template<class fun>
to_agg_index_t create_multiply(fun fn,
                               size_t split,
                               size_t num_nodes,
                               index_t a_mat,
                               index_t b_mat,
                               int32_t &tid_offset,
                               bbts::parsed_command_list_t &_cmds,
                               std::vector<std::vector<int32_t>> &to_del,
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

        // get the target node
        auto target_node = (node_id_t) fn(i, k, num_nodes);

        // add the command
        _cmds.add_apply("matrix_mult",
                        {"dense", "dense"},
                        {"dense"},
                        gpu_mult,
                        {{a_tid, target_node}, {b_tid, target_node}},
                        {{tid_offset, target_node}}, {});

        // mark that we need to delete this tensor later
        to_del[target_node].push_back(tid_offset);

        // do the multiplies
        multiplies[{i, j}].push_back({tid_offset, target_node});

        // go to next tid
        tid_offset++;
      }
    }
  }

  return std::move(multiplies);
}

void generate_aggregation(size_t split,
                          size_t num_nodes,
                          int32_t &tid_offset,
                          to_agg_index_t &multiplies,
                          bbts::parsed_command_list_t &_cmds,
                          bool gpu_add) {

  // create the aggregate
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // all the multiplied tensors
      auto &muls = multiplies[{rowID, colID}];

      // get the target node
      auto target_node = (node_id_t) ((rowID + colID * split) % num_nodes);

      // figure out the inputs
      std::vector<std::tuple<tid_t, node_id_t>> inputs;
      for (auto &mul : muls) {
        auto &[tid, node] = mul;
        inputs.emplace_back(tid, target_node);
      }

      // create the reduce command
      _cmds.add_reduce("matrix_add",
                       {"dense", "dense"},
                       {"dense"},
                       gpu_add,
                       inputs,
                       {tid_offset, target_node}, {});

      tid_offset++;
    }
  }
}

void create_delete(size_t num_nodes,
                   std::vector<std::vector<int32_t>> &to_del,
                   bbts::parsed_command_list_t &_cmds) {

  // prepare the removes
  for (int32_t node = 0; node < num_nodes; ++node) {

    // store the number we need to delete
    std::vector<std::tuple<tid_t, node_id_t>> _inputs;
    _inputs.reserve(to_del[node].size());
    for (auto t : to_del[node]) {
      _inputs.emplace_back(t, node);
    }

    // remove them from node
    _cmds.add_delete(_inputs);
  }
}

bbts::parsed_command_list_t generate_commands(bool gpu_add, bool gpu_mult, size_t split, size_t num_nodes, size_t matrix_size) {

  bbts::parsed_command_list_t commands;
  int32_t tid_offset = 0;

  auto a_idx = create_matrix_tensors(num_nodes, matrix_size, split, tid_offset, commands);
  auto b_idx = create_matrix_tensors(num_nodes, matrix_size, split, tid_offset, commands);

  // all the tensors that we need to delete
  std::vector<std::vector<int32_t>> to_del(num_nodes);

  // create the shuffle
  create_shuffle(num_nodes,
                 split,
                 [](int32_t rowID, int32_t colID, size_t num_nodes) { return rowID % num_nodes; },
                 a_idx,
                 commands,
                 to_del);

  // create the broadcast
  create_broadcast(num_nodes, split, b_idx, commands, to_del);

  // create the multiply commands
  auto multiplies = create_multiply([](int32_t rowID, int32_t colID, size_t num_nodes) { return rowID % num_nodes; },
                                    split, num_nodes,
                                    a_idx, b_idx, tid_offset, commands, to_del, gpu_mult);

  // generate the aggregation
  generate_aggregation(split, num_nodes, tid_offset, multiplies, commands, gpu_add);

  // create the delete
  create_delete(num_nodes, to_del, commands);

  return std::move(commands);
}

int main(int argc, char **argv) {

  if (argc != 7) {
    std::cout << "Incorrect usage\n";
    std::cout << "Usage ./generate_bmm <gpu_add> <gpu_mult> <split> <num_nodes> <matrix_size> <file>.bbts\n";
    return 0;
  }

  // get the parameters
  bool gpu_add = std::string(argv[1]) == "true" ? true : false;
  bool gpu_mult = std::string(argv[2]) == "true" ? true : false;
  
  // 
  char *end;
  auto split = std::strtol(argv[3], &end, 10);
  auto num_nodes = std::strtol(argv[4], &end, 10);
  auto matrix_size = std::strtol(argv[5], &end, 10);

  // store them
  auto t = generate_commands(gpu_add, gpu_mult, split, num_nodes, matrix_size);
  t.serialize(argv[6]);

  return 0;
}