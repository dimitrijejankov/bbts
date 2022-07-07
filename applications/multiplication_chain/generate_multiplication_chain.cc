
#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/parsed_command.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"
#include <cstdint>
#include <map>

using namespace bbts;

tid_t current_tid = 0;

const int32_t UNFORM_ID = 0;
const int32_t ADD_ID = 1;
const int32_t MUL_ID = 2;

void generate_matrix(int32_t num_row, int32_t num_cols, int32_t row_split,
                     int32_t col_spilt,
                     bbts::parsed_command_list_t &cmds,
                     std::map<std::tuple<int32_t, int32_t>, tid_t> &index,
                     int32_t node_start, int32_t node_end) {
  
  int32_t node_range = node_end - node_start;
  node_id_t node = 0;
  for (auto row_id = 0; row_id < row_split; row_id++) {
    for (auto col_id = 0; col_id < col_spilt; col_id++) {

      index[{row_id, col_id}] = current_tid;
      node = (node + 1) % node_range;

      // store the command
      cmds.add_apply("uniform",
                      {},
                      {"dense"},
                      false,
                      {},
                      {{current_tid++, node_start + node}},
                      {command_param_t{.u = (std::uint32_t)(num_row / row_split)},
                       command_param_t{.u = (std::uint32_t)(num_cols / row_split)},
                       command_param_t{.f = 0.0f},
                       command_param_t{.f = 1.0f}});
    }
  }
}

void generate_mult(
    int32_t n_split, int32_t m_spilt, int32_t k_split,
    std::vector<abstract_command_t> &commands,
    std::map<std::tuple<int32_t, int32_t>, tid_t> &a_index,
    std::map<std::tuple<int32_t, int32_t>, tid_t> &b_index,
    std::map<std::tuple<int32_t, int32_t>, std::vector<tid_t>> &c_tmp_index) {

  std::vector<command_param_t> param_data = {};

  for (auto n_id = 0; n_id < n_split; n_id++) {
    for (auto m_id = 0; m_id < m_spilt; m_id++) {
      for (auto k_id = 0; k_id < k_split; k_id++) {

        // store the c-idx
        c_tmp_index[{n_id, m_id}].push_back(current_tid);

        auto a = a_index[{n_id, k_id}];
        auto b = b_index[{k_id, m_id}];

        // store the command
        commands.push_back(abstract_command_t{
            .ud_id = MUL_ID,
            .type = abstract_command_type_t::APPLY,
            .input_tids = {a, b},
            .output_tids = {current_tid++},
            .params = param_data});
      }
    }
  }
}

void generate_sum(
    int32_t n_split, int32_t m_spilt, std::vector<abstract_command_t> &commands,
    std::map<std::tuple<int32_t, int32_t>, std::vector<tid_t>> &c_tmp_index,
    std::map<std::tuple<int32_t, int32_t>, tid_t> &out_index) {

  std::vector<command_param_t> param_data = {};
  for (auto n_id = 0; n_id < n_split; n_id++) {
    for (auto m_id = 0; m_id < m_spilt; m_id++) {

      auto c = c_tmp_index[{n_id, m_id}];

      // set the current tid
      out_index[{n_id, m_id}] = current_tid;

      // add the new reduce commands
      commands.push_back(
          abstract_command_t{.ud_id = ADD_ID,
                             .type = abstract_command_type_t::REDUCE,
                             .input_tids = c,
                             .output_tids = {current_tid++},
                             .params = {}});
    }
  }
}

void multiply(int32_t n_split, int32_t m_spilt, int32_t k_split,
              std::vector<abstract_command_t> &commands,
              std::map<std::tuple<int32_t, int32_t>, tid_t> &a_index,
              std::map<std::tuple<int32_t, int32_t>, tid_t> &b_index,
              std::map<std::tuple<int32_t, int32_t>, tid_t> &c_index) {

  std::map<std::tuple<int32_t, int32_t>, std::vector<tid_t>> c_tmp_index;
  generate_mult(n_split, m_spilt, k_split, commands, a_index, b_index,
                c_tmp_index);
  generate_sum(n_split, m_spilt, commands, c_tmp_index, c_index);
}

int main() {

  int32_t num_nodes;
  int32_t n, m, k, l, p;
  int32_t n_split, m_split, k_split, l_split, p_split;

  // ([n, m] x [m, k]) * ([k, l] x [l, p])
  std::cout << "the number of nodes : \n";
  std::cin >> num_nodes;
  std::cout << "([n, m] x [m, k]) * ([k, l] x [l, p])\n";
  std::cout << "n :";
  std::cin >> n;
  std::cout << "n_split :";
  std::cin >> n_split;
  std::cout << "m :";
  std::cin >> m;
  std::cout << "m_split :";
  std::cin >> m_split;
  std::cout << "k :";
  std::cin >> k;
  std::cout << "k_split :";
  std::cin >> k_split;
  std::cout << "l :";
  std::cin >> l;
  std::cout << "l_split :";
  std::cin >> l_split;
  std::cout << "p :";
  std::cin >> p;
  std::cout << "p_split :";
  std::cin >> p_split;

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // specify functions
  funs.push_back(abstract_ud_spec_t{.id = UNFORM_ID,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{.id = ADD_ID,
                                    .ud_name = "matrix_add",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  funs.push_back(abstract_ud_spec_t{.id = MUL_ID,
                                    .ud_name = "matrix_mult",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  // commands
  std::map<std::tuple<int32_t, int32_t>, tid_t> a_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> b_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> ab_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> c_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> d_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> cd_index;
  std::map<std::tuple<int32_t, int32_t>, tid_t> out_index;

  {
    bbts::parsed_command_list_t cmds;
    generate_matrix(n, m, n_split, m_split, cmds, a_index, 0, num_nodes / 2);
    generate_matrix(m, k, m_split, k_split, cmds, b_index, 0, num_nodes / 2);

    generate_matrix(k, l, k_split, l_split, cmds, c_index, num_nodes / 2, num_nodes);
    generate_matrix(l, p, l_split, p_split, cmds, d_index, num_nodes / 2, num_nodes);

    // write out the commands
    cmds.serialize("gen_partitioned.bbts");
  }

  {
    bbts::parsed_command_list_t cmds;
    current_tid = 0;
    generate_matrix(n, m, n_split, m_split, cmds, a_index, 0, num_nodes);
    generate_matrix(m, k, m_split, k_split, cmds, b_index, 0, num_nodes);

    generate_matrix(k, l, k_split, l_split, cmds, c_index, 0, num_nodes);
    generate_matrix(l, p, l_split, p_split, cmds, d_index, 0, num_nodes);

    // write out the commands
    cmds.serialize("gen_round_robin.bbts");
  }

  {
    // ([n, m] x [m, k]) * ([k, l] x [l, p])
    std::vector<abstract_command_t> commands;
    multiply(n_split, k_split, m_split, commands, a_index, b_index, ab_index);
    multiply(k_split, p_split, l_split, commands, c_index, d_index, cd_index);
    multiply(n_split, p_split, k_split, commands, ab_index, cd_index,
             out_index);

    // write out the commands
    std::ofstream gen("run.sbbts");
    compile_source_file_t gsf{.function_specs = funs, .commands = commands};
    gsf.write_to_file(gen);
    gen.close();
  }

  return 0;
}