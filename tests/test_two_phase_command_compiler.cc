#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../src/commands/command.h"
#include "../src/tensor/tensor.h"
#include "../src/tensor/tensor_factory.h"
#include "../src/ud_functions/ud_function.h"
#include "../src/ud_functions/udf_manager.h"
#include "../src/commands/two_layer_compiler.h"
#include "../src/commands/compile_source_file.h"


namespace bbts {

TEST(TestCommandCompiler, Test1) {

  // create the tensor factory
  auto factory = std::make_shared<tensor_factory_t>();

  // crate the udf manager
  auto manager = std::make_shared<udf_manager_t>(factory, nullptr);

  // the meta data
  std::unordered_map<tid_t, tensor_meta_t> meta;

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // matrix addition
  funs.push_back(abstract_ud_spec_t{.id = 0,
                                    .ud_name = "matrix_add",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  // matrix multiplication
  funs.push_back(abstract_ud_spec_t{.id = 1,
                                    .ud_name = "matrix_mult",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});
  
  // the uniform distribution
  funs.push_back(abstract_ud_spec_t{.id = 2,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  // init the cost model
  auto cost_model = std::make_shared<cost_model_t>(meta,
                                                   funs,
                                                   factory, 
                                                   manager, 
                                                   1.0f,
                                                   1.0f);

  // init the compiler
  auto compiler = std::make_shared<two_layer_compiler>(cost_model, 2);

  std::vector<command_param_t> param_data = {command_param_t{.u = 100},
                                             command_param_t{.u = 100},
                                             command_param_t{.f = 0.0f},
                                             command_param_t{.f = 1.0f}};

  // the commands
  std::vector<abstract_command_t> commands = {

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {0}, // A(0, 0)
                       .params = param_data},
                       
    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {1}, // A(0, 1)
                       .params = param_data},

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {2}, // A(1, 0)
                       .params = param_data},

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {3}, // A(1, 1)
                       .params = param_data},

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {4}, // B(0, 0)
                       .params = param_data},
                       
    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {5}, // B(0, 1)
                       .params = param_data},

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {6}, // B(1, 0)
                       .params = param_data},

    abstract_command_t{.ud_id = 2,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {}, 
                       .output_tids = {7}, // B(1, 1)
                       .params = param_data}
  };

  // compile them
  std::vector<std::unordered_set<tid_t>> tensor_locations(2); 
  auto cmds = compiler->compile(commands, tensor_locations);

  // print out all the location
  for(node_id_t node = 0; node < 2; node++) {
    std::cout << "Node : " << node << '\n';
    for(auto ts : tensor_locations[node]) {
      std::cout << "tid : " << ts << '\n';
    }
  }

  // the commands
  commands = {

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {0, 4}, // A(0, 0) B(0, 0)
                       .output_tids = {8}, // C(0, 0)
                       .params = {}},
                       
    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {1, 6}, // A(0, 1) B(1, 0)
                       .output_tids = {9}, // C(0, 0)
                       .params = {}},

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {2, 4}, // A(1, 0) B(0, 0)
                       .output_tids = {10}, // C(1, 0)
                       .params = {}},

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {3, 6}, // A(1, 1) B(1, 0)
                       .output_tids = {11}, // C(1, 0)
                       .params = {}},

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {0, 5}, // A(0, 0) B(0, 1)
                       .output_tids = {12}, // C(0, 1)
                       .params = {}},
                       
    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {1, 7}, // A(0, 1) B(1, 1)
                       .output_tids = {13}, // C(0, 1)
                       .params = {}},

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {2, 5}, // A(1, 0) B(0, 1)
                       .output_tids = {14}, // C(1, 1)
                       .params = {}},

    abstract_command_t{.ud_id = 1,
                       .type = abstract_command_type_t::APPLY,
                       .input_tids = {3, 7}, // A(1, 1) B(1, 1)
                       .output_tids = {15}, // C(1, 1)
                       .params = {}},

    abstract_command_t{.ud_id = 0,
                       .type = abstract_command_type_t::REDUCE,
                       .input_tids = {8, 9}, // A(0, 0) B(0, 1)
                       .output_tids = {16}, // C(0, 1)
                       .params = {}},

    abstract_command_t{.ud_id = -1,
                       .type = abstract_command_type_t::DELETE,
                       .input_tids = {8, 9}, // A(1, 1), B(1, 1)
                       .output_tids = {},
                       .params = {}},
                       
    abstract_command_t{.ud_id = 0,
                       .type = abstract_command_type_t::REDUCE,
                       .input_tids = {10, 11}, // A(0, 1) B(1, 1)
                       .output_tids = {17}, // C(0, 1)
                       .params = {}},

    abstract_command_t{.ud_id = -1,
                       .type = abstract_command_type_t::DELETE,
                       .input_tids = {10, 11}, // A(0, 1) B(1, 1)
                       .output_tids = {},
                       .params = {}},

    abstract_command_t{.ud_id = 0,
                       .type = abstract_command_type_t::REDUCE,
                       .input_tids = {12, 13}, // A(1, 0) B(0, 1)
                       .output_tids = {18}, // C(1, 1)
                       .params = {}},

    abstract_command_t{.ud_id = -1,
                       .type = abstract_command_type_t::DELETE,
                       .input_tids = {12, 13}, // A(1, 0) B(0, 1)
                       .output_tids = {},
                       .params = {}},

    abstract_command_t{.ud_id = 0,
                       .type = abstract_command_type_t::REDUCE,
                       .input_tids = {14, 15}, // A(1, 1) B(1, 1)
                       .output_tids = {19}, // C(1, 1)
                       .params = {}},
      
    abstract_command_t{.ud_id = -1,
                       .type = abstract_command_type_t::DELETE,
                       .input_tids = {14, 15}, // A(1, 1) B(1, 1)
                       .output_tids = {},
                       .params = {}}

  };

  cmds = compiler->compile(commands, tensor_locations);

  std::stringstream ss;
  for(auto &c : cmds) {
    c->print(ss);
  }
  std::cout << ss.str() << '\n'; 

  // print out all the location
  for(node_id_t node = 0; node < 2; node++) {
    std::cout << "Node : " << node << '\n';
    for(auto ts : tensor_locations[node]) {
      std::cout << "tid : " << ts << '\n';
    }
  }

}


TEST(TestCommandCompiler, Test2) {

  const int32_t num_nodes = 4; 

  // create the tensor factory
  auto factory = std::make_shared<tensor_factory_t>();

  // crate the udf manager
  auto manager = std::make_shared<udf_manager_t>(factory, nullptr);

  // the parameters
  std::vector<command_param_t> param_data = {command_param_t{.u = 100},
                                             command_param_t{.u = 100},
                                             command_param_t{.f = 0.0f},
                                             command_param_t{.f = 1.0f}};

  // the initial indices
  tid_t curID = 0;
  std::vector<abstract_command_t> commands;
  std::map<std::tuple<int32_t, int32_t>, tid_t> a_idx;
  for(int32_t idx = 0; idx < num_nodes; ++idx) {
    for(int32_t jdx = 0; jdx < num_nodes; ++jdx) {
      
      commands.push_back(abstract_command_t{.ud_id = 2,
                                            .type = abstract_command_type_t::APPLY,
                                            .input_tids = {}, 
                                            .output_tids = {curID}, // A(idx, jdx)
                                            .params = param_data});

      a_idx[{idx, jdx}] = curID++;
    }
  }

  // the initial indices
  std::map<std::tuple<int32_t, int32_t>, tid_t> b_idx;
  for(int32_t idx = 0; idx < num_nodes; ++idx) {
    for(int32_t jdx = 0; jdx < num_nodes; ++jdx) {

      commands.push_back(abstract_command_t{.ud_id = 2,
                                            .type = abstract_command_type_t::APPLY,
                                            .input_tids = {}, 
                                            .output_tids = {curID}, // B(idx, jdx)
                                            .params = param_data});

      b_idx[{idx, jdx}] = curID++;
    }
  }

  // the meta data
  std::unordered_map<tid_t, tensor_meta_t> meta;

  // the functions
  std::vector<abstract_ud_spec_t> funs;

  // matrix addition
  funs.push_back(abstract_ud_spec_t{.id = 0,
                                    .ud_name = "matrix_add",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});

  // matrix multiplication
  funs.push_back(abstract_ud_spec_t{.id = 1,
                                    .ud_name = "matrix_mult",
                                    .input_types = {"dense", "dense"},
                                    .output_types = {"dense"}});
  
  // the uniform distribution
  funs.push_back(abstract_ud_spec_t{.id = 2,
                                    .ud_name = "uniform",
                                    .input_types = {},
                                    .output_types = {"dense"}});

  // init the cost model
  auto cost_model = std::make_shared<cost_model_t>(meta,
                                                   funs,
                                                   factory, 
                                                   manager, 
                                                   1.0f,
                                                   1.0f);

  // init the compiler
  auto compiler = std::make_shared<two_layer_compiler>(cost_model, num_nodes);
  
  std::ofstream f("a");
  compile_source_file_t tmp1 {.function_specs = funs, .commands = commands };
  tmp1.write_to_file(f);
  f.close();

  // compile them
  std::vector<std::unordered_set<tid_t>> tensor_locations(num_nodes); 
  auto cmds = compiler->compile(commands, tensor_locations);

  std::stringstream ss;
  for(auto &c : cmds) {
    c->print(ss);
  }
  std::cout << ss.str() << '\n';

  // make the next batch of commands
  commands.clear();

  // make the multiples
  std::map<std::tuple<int32_t, int32_t>, std::vector<tid_t>> c_idx;
  for(int32_t idx = 0; idx < num_nodes; ++idx) {
    for(int32_t jdx = 0; jdx < num_nodes; ++jdx) {
      for(int32_t kdx = 0; kdx < num_nodes; ++kdx) {

        // 
        commands.push_back(abstract_command_t{.ud_id = 1,
                                              .type = abstract_command_type_t::APPLY,
                                              .input_tids = {a_idx[{idx, kdx}], b_idx[{kdx, jdx}]}, // A(idx, kdx) B(kdx, jdx)
                                              .output_tids = {curID}, // C(idx, jdx)
                                              .params = {}});

        // 
        c_idx[{idx, jdx}].push_back(curID++);
      }
    }
  }

  // make the reduce ops
  for(auto &c : c_idx) {

    // add the new reduce commands
    commands.push_back(abstract_command_t{.ud_id = 0,
                                          .type = abstract_command_type_t::REDUCE,
                                          .input_tids = c.second, 
                                          .output_tids = {curID++},
                                          .params = {}});

    // add the delete to remove the intermediate
    commands.push_back(abstract_command_t{.ud_id = -1,
                                          .type = abstract_command_type_t::DELETE,
                                          .input_tids =  c.second,
                                          .output_tids = {},
                                          .params = {}});
  }

  cmds = compiler->compile(commands, tensor_locations);

  std::ofstream f1("b");
  compile_source_file_t tmp2 {.function_specs = funs, .commands = commands };
  tmp2.write_to_file(f1);
  f1.close();

  std::stringstream ss2;
  for(auto &c : cmds) {
    c->print(ss2);
  }
  std::cout << ss2.str() << '\n';

}

}