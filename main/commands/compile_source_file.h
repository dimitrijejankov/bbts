#pragma once

#include <vector>
#include "abstract_command.h"

namespace bbts {

struct compile_source_file_t {

  // specifies all the functions used
  std::vector<abstract_ud_spec_t> function_specs;

  // specifies all the commands used
  std::vector<abstract_command_t> commands;

  void write_to_file(std::ofstream &file) {

    // write number of functions
    file << function_specs.size() << "\n";

    // write the function specs
    for(auto &fn : function_specs) {
      fn.write_to_file(file);
    }

    // write the command numbers
    file << commands.size() << '\n';

    // write the commands
    for(auto &cmd : commands) {
      cmd.write_to_file(file);
    }
  }

  void read_from_file(std::ifstream &file) {

    // get the number of functions
    std::size_t num_funs;
    file >> num_funs;

    // read all the functions
    function_specs.resize(num_funs);
    for(auto &fn : function_specs) {
      fn.read_from_file(file);
    }

    // read the number of commands
    std::size_t num_cmds;
    file >> num_cmds;

    // read all the commands
    commands.resize(num_cmds);
    for(auto &cmd : commands) {
      cmd.read_from_file(file);
    }
  }

};


}