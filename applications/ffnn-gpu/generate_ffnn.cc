#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <utility>
#include <vector>

#include "../../src/commands/compile_source_file.h"
#include "../../src/commands/two_layer_compiler.h"
#include "../../src/tensor/tensor.h"
#include "ffnn_add.h"

const int32_t FFNN_ACT_MULT = 0;
const int32_t FFNN_ADD = 1;
const int32_t FFNN_MATRIX_HADAMARD = 2;
const int32_t FFNN_MULT = 3;
const int32_t FFNN_UNIFORM_DATA = 4;
const int32_t FFNN_UNIFORM_WEIGHTS = 5;
const int32_t FFNN_WEIGHTED_SUM = 6;
const int32_t FFNN_MULT_BACK = 7;
const int32_t FFNN_UNIFORM_SPARSE = 8;
const int32_t FFNN_WEIGHTED_SUM_SPARSE_DENSE = 9;

using namespace bbts;

float learning_rate = 1.0f;

int32_t num_nodes;

int32_t num_batch;
int32_t batch_block;

int32_t num_features;
int32_t features_block;

int32_t num_labels;
int32_t labels_block;

int32_t embedding_size;
int32_t embedding_block;

bbts::tid_t currentTID = 0;

using matrix_index = std::map<std::tuple<int32_t, int32_t>, bbts::tid_t>;
using matrix_reduce_index =
    std::map<std::tuple<int32_t, int32_t>, std::vector<bbts::tid_t>>;

void remove_matrix(const matrix_index &idx,
                   std::vector<abstract_command_t> &commands) {

  // make all the removes
  std::vector<tid_t> to_remove;
  for (auto it : idx) {
    to_remove.push_back(it.second);
  }

  // add the delete to remove the intermediate
  commands.push_back(abstract_command_t{.ud_id = -1,
                                        .type = abstract_command_type_t::DELETE,
                                        .input_tids = std::move(to_remove),
                                        .output_tids = {},
                                        .params = {}});
}

matrix_index generate_random(abstract_ud_spec_id_t ud,
                             std::vector<abstract_command_t> &commands,
                             size_t num_rows, size_t num_cols,
                             size_t split_rows, size_t split_cols) {

  // the parameter data
  std::vector<command_param_t> param_data = {
      command_param_t{.i = (std::int32_t)(num_rows / split_rows)},
      command_param_t{.i = (std::int32_t)(num_cols / split_cols)},
      command_param_t{.i = 0},
      command_param_t{.i = 0},
      command_param_t{.f = 0.0f}, 
      command_param_t{.f = 1.0f}};

  matrix_index out;
  for (size_t rowID = 0; rowID < split_rows; ++rowID) {
    for (size_t colID = 0; colID < split_cols; ++colID) {

      // set the row and column id
      param_data[2].i = rowID;
      param_data[3].i = colID;

      // store the index
      auto tid = currentTID++;
      out[{rowID, colID}] = tid;

      // store the command
      commands.push_back(
          abstract_command_t{.ud_id = ud,
                             .type = abstract_command_type_t::APPLY,
                             .input_tids = {},
                             .output_tids = {tid},
                             .params = param_data});
    }
  }

  return std::move(out);
}

matrix_index apply_unary(abstract_ud_spec_id_t ud,
                         std::vector<abstract_command_t> &commands,
                         matrix_index &in,
                         const std::vector<command_param_t> &param_data) {

  matrix_index out;
  for (auto r : in) {

    auto tid = currentTID++;
    out[r.first] = tid;

    commands.push_back(
        abstract_command_t{.ud_id = ud,
                           .type = abstract_command_type_t::APPLY,
                           .input_tids = {r.second},
                           .output_tids = {tid},
                           .params = param_data});
  }

  return std::move(out);
}

matrix_index apply_binary(abstract_ud_spec_id_t ud,
                          std::vector<abstract_command_t> &commands,
                          matrix_index &lhs, matrix_index &rhs,
                          const std::vector<command_param_t> &param_data) {

  matrix_index out;
  for (auto l : lhs) {

    assert(rhs.find(l.first) != rhs.end());
    auto r = rhs[l.first];

    auto tid = currentTID++;
    out[l.first] = tid;

    commands.push_back(
        abstract_command_t{.ud_id = ud,
                           .type = abstract_command_type_t::APPLY,
                           .input_tids = {l.second, r},
                           .output_tids = {tid},
                           .params = param_data});
  }

  return std::move(out);
}

matrix_index generate_multiply(abstract_ud_spec_id_t ud,
                               std::vector<abstract_command_t> &commands,
                               matrix_index &lhs, matrix_index &rhs,
                               bool lhs_trans, bool rhs_trans, int32_t n,
                               int32_t m, int32_t k,
                               ffnn_add::elementwise_fn_type final_op) {

  // the parameter data
  std::vector<command_param_t> param_data = {command_param_t{.b = lhs_trans},
                                             command_param_t{.b = rhs_trans}};
  // make the multiplies
  matrix_reduce_index ridx;
  for (int32_t ni = 0; ni < n; ni++) {
    for (int32_t mi = 0; mi < m; mi++) {
      for (int32_t ki = 0; ki < k; ki++) {

        // get the right row and column id from the left matrix
        auto l_row = lhs_trans ? ki : ni;
        auto l_col = lhs_trans ? ni : ki;

        // get the right row and column id from the right matrix
        auto r_row = rhs_trans ? mi : ki;
        auto r_col = rhs_trans ? ki : mi;

        // store it
        auto tid = currentTID++;
        ridx[{ni, mi}].push_back(tid);

        assert(lhs.find({l_row, l_col}) != lhs.end());
        assert(rhs.find({r_row, r_col}) != rhs.end());

        // make the command
        commands.push_back(abstract_command_t{
            .ud_id = ud,
            .type = abstract_command_type_t::APPLY,
            .input_tids = {lhs[{l_row, l_col}], rhs[{r_row, r_col}]},
            .output_tids = {tid},
            .params = param_data});
      }
    }
  }

  param_data = {command_param_t{.i = k},
                command_param_t{.i = (int32_t)final_op}};

  // make the reduce ops
  matrix_index out;
  for (auto &c : ridx) {

    // set the current tid
    out[c.first] = currentTID;

    // add the new reduce commands
    commands.push_back(
        abstract_command_t{.ud_id = FFNN_ADD,
                           .type = abstract_command_type_t::REDUCE,
                           .input_tids = c.second,
                           .output_tids = {currentTID++},
                           .params = param_data});

    // add the delete to remove the intermediate
    commands.push_back(
        abstract_command_t{.ud_id = -1,
                           .type = abstract_command_type_t::DELETE,
                           .input_tids = c.second,
                           .output_tids = {},
                           .params = {}});
  }

  // return the idex
  return std::move(out);
}

int main(int argc, char **argv) {

  std::cout << "num_nodes : \n";
  std::cin >> num_nodes;

  std::cout << "num_batch : \n";
  std::cin >> num_batch;

  std::cout << "batch_block : \n";
  std::cin >> batch_block;

  std::cout << "num_features : \n";
  std::cin >> num_features;

  std::cout << "features_block : \n";
  std::cin >> features_block;

  std::cout << "num_labels : \n";
  std::cin >> num_labels;

  std::cout << "labels_block : \n";
  std::cin >> labels_block;

  std::cout << "embedding_size : \n";
  std::cin >> embedding_size;

  std::cout << "embedding_block : \n";
  std::cin >> embedding_block;

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  funs.push_back(abstract_ud_spec_t{.id = FFNN_ACT_MULT,
                                    .ud_name = "ffnn_act_mult",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_ADD,
                                    .ud_name = "ffnn_add",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_MATRIX_HADAMARD,
                                    .ud_name = "ffnn_matrix_hadamard",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_MULT,
                                    .ud_name = "ffnn_mult",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_UNIFORM_DATA,
                                    .ud_name = "ffnn_uniform_data",
                                    .input_types = {},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_UNIFORM_WEIGHTS,
                                    .ud_name = "ffnn_uniform_weights",
                                    .input_types = {},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_WEIGHTED_SUM,
                                    .ud_name = "ffnn_weighted_sum",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_MULT_BACK,
                                    .ud_name = "ffnn_back_mult",
                                    .input_types = {"ffnn_dense", "ffnn_dense"},
                                    .output_types = {"ffnn_dense"}});

  funs.push_back(abstract_ud_spec_t{.id = FFNN_UNIFORM_SPARSE,
                                    .ud_name = "ffnn_uniform_sparse_data",
                                    .input_types = {},
                                    .output_types = {"ffnn_sparse"}});

  funs.push_back(
      abstract_ud_spec_t{.id = FFNN_WEIGHTED_SUM_SPARSE_DENSE,
                         .ud_name = "ffnn_weighted_sum_sparse_dense",
                         .input_types = {"ffnn_sparse", "ffnn_dense"},
                         .output_types = {"ffnn_dense"}});

  // generate the matrices
  std::vector<abstract_command_t> generate_matrices;

  // generate the input batch
  auto x = generate_random(FFNN_UNIFORM_DATA, generate_matrices, num_batch,
                           num_features, num_batch / batch_block,
                           num_features / features_block);
  auto y = generate_random(FFNN_UNIFORM_SPARSE, generate_matrices, num_batch,
                           num_labels, num_batch / batch_block,
                           num_labels / labels_block);

  // init the weights
  auto w1 = generate_random(
      FFNN_UNIFORM_WEIGHTS, generate_matrices, num_features, embedding_size,
      num_features / features_block, embedding_size / embedding_block);
  auto w2 = generate_random(
      FFNN_UNIFORM_WEIGHTS, generate_matrices, embedding_size, num_labels,
      embedding_size / embedding_block, num_labels / labels_block);

  // write out the commands
  std::ofstream gen("gen.sbbts");
  compile_source_file_t gsf{.function_specs = funs,
                            .commands = generate_matrices};
  gsf.write_to_file(gen);
  gen.close();

  std::vector<abstract_command_t> ffnn_commands;

  // a_1 = relu(X * W1 + b)
  auto a_1 = generate_multiply(
      FFNN_ACT_MULT, ffnn_commands, x, w1, false, false,
      num_batch / batch_block, embedding_size / embedding_block,
      num_features / features_block, ffnn_add::elementwise_fn_type::RELU);

  // a_2 = sigmoid(a_1 * W2 + b)
  auto a_2 = generate_multiply(
      FFNN_ACT_MULT, ffnn_commands, a_1, w2, false, false,
      num_batch / batch_block, num_labels / labels_block,
      embedding_size / embedding_block, ffnn_add::elementwise_fn_type::SIGMOID);

  // ∇a_2 = a2 − Y
  std::vector<command_param_t> param_data = {command_param_t{.f = -1.0f},
                                             command_param_t{.f = 1.0f}};
  auto delta_a_2 =
      apply_binary(FFNN_WEIGHTED_SUM_SPARSE_DENSE, ffnn_commands, y, a_2, {});

  // ∇w_2 = a_1^𝑇 * ∇a_2
  auto delta_w_2 = generate_multiply(
      FFNN_MULT_BACK, ffnn_commands, a_1, delta_a_2, true, false,
      embedding_size / embedding_block, num_labels / labels_block,
      num_batch / batch_block, ffnn_add::elementwise_fn_type::NOOP);

  // ∇a_2 * W_2^𝑇
  auto delta_a_1_tmp = generate_multiply(
      FFNN_MULT, ffnn_commands, delta_a_2, w2, false, true,
      num_batch / batch_block, embedding_size / embedding_block,
      num_labels / labels_block, ffnn_add::elementwise_fn_type::NOOP);

  // ∇a_1 = ∇a_2 * W_2^𝑇 .* relu'(a1)
  auto delta_a_1 = apply_binary(FFNN_MATRIX_HADAMARD, ffnn_commands,
                                delta_a_1_tmp, a_1, {});

  // ∇w_1 = x^𝑇 * ∇a_1
  auto delta_w_1 = generate_multiply(
      FFNN_MULT_BACK, ffnn_commands, x, delta_a_1, true, false,
      num_features / features_block, embedding_size / embedding_block,
      num_batch / batch_block, ffnn_add::elementwise_fn_type::NOOP);

  // update the weights
  param_data = {command_param_t{.f = -learning_rate}};
  auto updated_w1 =
      apply_binary(FFNN_WEIGHTED_SUM, ffnn_commands, w1, delta_w_1, {});
  auto updated_w2 =
      apply_binary(FFNN_WEIGHTED_SUM, ffnn_commands, w2, delta_w_2, {});

  // do a ton of removes
  remove_matrix(w1, ffnn_commands);
  remove_matrix(w2, ffnn_commands);
  remove_matrix(a_1, ffnn_commands);
  remove_matrix(a_2, ffnn_commands);
  remove_matrix(delta_a_2, ffnn_commands);
  remove_matrix(delta_w_2, ffnn_commands);
  remove_matrix(delta_a_1_tmp, ffnn_commands);
  remove_matrix(delta_a_1, ffnn_commands);
  remove_matrix(delta_w_1, ffnn_commands);

  // write the generated
  std::ofstream gen2("run.sbbts");
  compile_source_file_t gsf2{.function_specs = funs, .commands = ffnn_commands};
  gsf2.write_to_file(gen2);
  gen2.close();

  return 0;
}