#include <map>
#include <thread>
#include "../src/operations/move_op.h"
#include "../src/commands/reservation_station.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
using namespace bbts;

using index_t = std::map<std::tuple<int, int>, int>;
using multi_index_t = std::map<std::tuple<int, int>, std::vector<int>>;

index_t create_matrix_tensors(reservation_station_ptr_t &rs, bbts::tensor_factory_ptr_t &tf, bbts::storage_ptr_t &ts,
                              int n, int split, int my_rank, int num_nodes, int &cur_tid) {

  // the index
  index_t index;

  // block size
  int block_size = n / split;

  // grab the format impl_id of the dense tensor
  auto fmt_id = tf->get_tensor_ftm("dense");

  // create all the rows an columns we need
  auto hash_fn = std::hash<int>();
  for(int row_id = 0; row_id < split; ++row_id) {
    for(int col_id = 0; col_id < split; ++col_id) {

      // check if this block is on this node
      auto hash = hash_fn(row_id * split + col_id) % num_nodes;
      if(hash == my_rank) {

        // ok it is on this node make a tensor
        // make the meta
        dense_tensor_meta_t dm{fmt_id, block_size, block_size};

        // get the size of the tensor we need to crate
        auto tensor_size = tf->get_tensor_size(dm);

        // crate the tensor
        auto t = ts->create_tensor(cur_tid, tensor_size);

        // init the tensor
        auto &dt = tf->init_tensor(t, dm).as<dense_tensor_t>();

        // set the index
        index[{row_id, col_id}] = cur_tid;
        rs->register_tensor(cur_tid);
      }

      // go to the next one
      cur_tid++;
    }
  }

  // return the index
  return std::move(index);
}

std::vector<command_ptr_t> create_broadcast(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

  // go through the tuples
  std::vector<command_ptr_t> commands;
  for(auto &t : idx) {

    // make the command
    auto cmd = std::make_unique<bbts::command_t>();
    cmd->_type = bbts::command_t::MOVE;
    cmd->_id = cur_cmd++;
    cmd->_input_tensors.push_back({.tid = t.second, .node = my_rank});

    // go through all the nodes
    for(int32_t node = 0; node < num_nodes; ++node) {

      // skip this node
      if(node == my_rank) { continue; }

      // set the output tensor
      cmd->_output_tensors.push_back({.tid = t.second, .node = my_rank});
    }

    // store the command
    commands.emplace_back(std::move(cmd));
  }

  return commands;
}

std::vector<command_ptr_t> create_shuffle(index_t &idx, int my_rank, int num_nodes, int &cur_cmd) {

  // go through the tuples
  std::vector<command_ptr_t> commands;
  for(auto &t : idx) {

    // where we need to move
    int32_t to_node = std::get<0>(t.first) % num_nodes;

    //
    if(to_node == my_rank) {
      continue;
    }

    // make the command
    auto cmd = std::make_unique<bbts::command_t>();
    cmd->_type = bbts::command_t::MOVE;
    cmd->_id = cur_cmd++;
    cmd->_input_tensors.push_back({.tid = t.second, .node = my_rank});
    cmd->_output_tensors.push_back({.tid = t.second, .node = to_node});

    // store the command
    commands.emplace_back(std::move(cmd));
  }

  return std::move(commands);
}

std::vector<command_ptr_t> create_join(udf_manager_ptr &udm, index_t &lhs, index_t &rhs, multi_index_t &out_idx,
                                       int my_rank, int split, int &cur_cmd, int &cur_tid) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_mult");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate all the commands
  std::vector<command_ptr_t> commands;
  for(int a_row_id = 0; a_row_id < split; ++a_row_id) {
    for (int b_col_id = 0; b_col_id < split; ++b_col_id) {

      // create all the join groups that need to be reduced together
      auto &tensor_to_reduce = out_idx[{a_row_id, b_col_id}];
      for (int ab_row_col_id = 0; ab_row_col_id < split; ++ab_row_col_id) {

        // make the command
        auto cmd = std::make_unique<bbts::command_t>();
        cmd->_type = bbts::command_t::APPLY;
        cmd->_id = cur_cmd++;
        cmd->_fun_id = ud->impl_id;

        // get the tids for the left and right
        auto l = lhs[{a_row_id, ab_row_col_id}];
        auto r = rhs[{ab_row_col_id, b_col_id}];

        // set the left and right input
        cmd->_input_tensors.push_back({.tid = l, .node = my_rank});
        cmd->_input_tensors.push_back({.tid = r, .node = my_rank});

        //set the output
        cmd->_output_tensors.push_back({.tid = cur_tid, .node = my_rank});

        // store the command
        commands.emplace_back(move(cmd));
        tensor_to_reduce.push_back(cur_tid++);
      }
    }
  }

  // move the commands
  return std::move(commands);
}

std::vector<command_ptr_t> create_agg(udf_manager_ptr &udm, multi_index_t &to_agg_idx, index_t &out,
                                      int my_rank, int &cur_cmd, int &cur_tid) {

  // return me that matcher for matrix addition
  auto matcher = udm->get_matcher_for("matrix_add");

  // get the ud object
  auto ud = matcher->findMatch({"dense", "dense"}, {"dense"}, false, 0);

  // generate all the commands
  std::vector<command_ptr_t> commands;
  for(auto &to_reduce : to_agg_idx) {

    // make the command
    auto cmd = std::make_unique<bbts::command_t>();
    cmd->_type = bbts::command_t::REDUCE;
    cmd->_id = cur_cmd++;
    cmd->_fun_id = ud->impl_id;

    // set the input tensors we want to reduce
    for(auto tid : to_reduce.second) {
      cmd->_input_tensors.push_back({.tid = tid, .node = my_rank});
    }

    cmd->_output_tensors.push_back({.tid = cur_tid++, .node = my_rank});
    commands.emplace_back(move(cmd));
  }

  return std::move(commands);
}

void schedule_all(bbts::reservation_station_t &rs, std::vector<command_ptr_t> &cmds) {

  // schedule the commands
  std::cout << "Scheduling " << cmds.size() << "\n";
  for(auto &c : cmds) {
    rs.queue_command(std::move(c));
  }
}

int main(int argc, char **argv) {

  // the number of threads per node
  const int32_t num_threads = 4;

  // make the configuration
  auto config = std::make_shared<bbts::node_config_t>(bbts::node_config_t{.argc=argc, .argv = argv});

  // create the storage
  storage_ptr_t ts = std::make_shared<storage_t>();

  // create the tensor factory
  auto tf = std::make_shared<bbts::tensor_factory_t>();

  // crate the udf manager
  auto udm = std::make_shared<udf_manager>(tf);

  // init the communicator with the configuration
  bbts::communicator_t comm(config);
  auto my_rank = comm.get_rank();
  auto num_nodes = comm.get_num_nodes();
  
  // create the reservation station
  auto rs = std::make_shared<bbts::reservation_station_t>(my_rank, ts);
  
  // create two tensors split into num_nodes x num_nodes, we split them by some hash
  std::cout << "Creating tensors....\n";
  int tid_offset = 0;
  auto a_idx = create_matrix_tensors(rs, tf, ts, 1000, num_nodes, my_rank, num_nodes, tid_offset);
  auto b_idx = create_matrix_tensors(rs, tf, ts, 1000, num_nodes, my_rank, num_nodes, tid_offset);

  // create the broadcast commands
  std::cout << "Creating broadcast commands...\n";
  int32_t cmd_offest = 0;
  auto bcast_cmds = create_broadcast(a_idx, my_rank, num_nodes, cmd_offest);

  // create the shuffle commands
  std::cout << "Create the shuffle commands...\n";
  auto shuffle_cmds = create_shuffle(b_idx, my_rank, num_nodes, cmd_offest);

  // create an join commands
  std::cout << "Creating join commands...\n";
  multi_index_t join;
  auto join_cmds = create_join(udm, a_idx, b_idx, join, my_rank, num_nodes, cmd_offest, tid_offset);

  // create an aggregation commands
  std::cout << "Creating aggregation commands...\n";
  index_t final;
  auto agg_cmds = create_agg(udm, join, final, my_rank, cmd_offest, tid_offset);

  // schedule the commands
  schedule_all(*rs, bcast_cmds);
  schedule_all(*rs, shuffle_cmds);
  schedule_all(*rs, join_cmds);
  schedule_all(*rs, agg_cmds);

  // kick of a bunch of threads that are going to grab commands
  std::vector<std::thread> commandExecutors;
  commandExecutors.reserve(num_threads);
  for(int32_t t = 0; t < num_threads; ++t) {

    // each thread is grabbing a command
    commandExecutors.emplace_back([&rs, &ts, &udm]() {

      std::vector<tensor_meta_t> _out_meta_tmp;

      for(;;) {

        // grab the next command
        auto cmd = rs->get_next_command();

        // are we doing an apply (applies are local so we are cool)
        if (cmd->_type == bbts::command_t::APPLY) {

          // get the ud function we want to run
          auto call_me = udm->get_fn_impl(cmd->_fun_id);

          // make the meta for the input
          std::vector<tensor_meta_t*> inputs_meta;
          for(auto &in : cmd->_input_tensors) { inputs_meta.push_back(&ts->get_by_tid(in.tid)->_meta);  }
          bbts::ud_impl_t::meta_params_t input_meta(move(inputs_meta));

          // get the meta for the outputs
          _out_meta_tmp.resize(cmd->_output_tensors.size());
          std::vector<tensor_meta_t*> outputs_meta;
          outputs_meta.reserve(_out_meta_tmp.size());

          // fill them up
          for(auto &om : _out_meta_tmp) { outputs_meta.push_back(&om);  }
          bbts::ud_impl_t::meta_params_t out_meta(std::move(outputs_meta));

          // get the meta
          call_me->get_out_meta(input_meta, out_meta);

          // form all the inputs
          std::vector<tensor_t*> inputs;
          for(auto &in : cmd->_input_tensors) { inputs.push_back(ts->get_by_tid(in.tid));  }
          ud_impl_t::tensor_params_t inputParams = {std::move(inputs)};

          // form the outputs
          std::vector<tensor_t*> outputs;
          for(auto &out : cmd->_output_tensors) {

            // get the size of tensor
            auto num_bytes = tf->get_tensor_size(out)

            outputs.push_back(ts->get_by_tid(out.tid));
          }
          ud_impl_t::tensor_params_t outputParams = {std::move(outputs)};

          // apply the function
          call_me->fn(inputParams, outputParams);

          std::cout << "Executed " << cmd->_type << "\n";
        }

      }

    });
  }

  // wait for the threads to finish
  for(auto &t : commandExecutors) {
    t.join();
  }

  return 0;
}
#pragma clang diagnostic pop