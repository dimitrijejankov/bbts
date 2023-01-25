#pragma once

#include "../ud_functions/udf_manager.h"
#include "../tensor/tensor_factory.h"
#include "../commands/command.h"
#include "../commands/parsed_command.h"

namespace bbts {

class command_loader_t {

 public:

  command_loader_t(tensor_factory_t &factory,
                     udf_manager_t &udf_manager) : _factory(factory),
                                                   _udf_manager(udf_manager) {}

  std::vector<command_ptr_t> compile(const bbts::parsed_command_list_t &cmds) {

    // the starting command
    command_id_t cur_cmd = 0;

    // compile each commands
    std::vector<command_ptr_t> compiled;
    for (size_t i = 0; i < cmds.get_num_commands(); ++i) {
      compiled.push_back(std::move(compile_cmd(cmds[i], cur_cmd)));
    }

    // return the compiled commands
    return std::move(compiled);
  }

 private:

  command_ptr_t compile_cmd(const parsed_command_t &cmd, command_id_t &cur_cmd) {

    switch (cmd.type) {

      case parsed_command_t::op_type_t::MOVE : {

        // make sure there is only one input
        if(cmd.inputs.size() != 1) {
          throw std::runtime_error("A move operation can only have one input.");
        }

        // make sure there is at least one output
        if(cmd.outputs.size() == 0) {
          throw std::runtime_error("A move operation must have at least one output.");
        }

        // is this a BROADCAST?
        if(cmd.outputs.size() != 1) {

          // no need to move here
          std::vector<command_t::tid_node_id_t> outputs;
          for(auto &out : cmd.outputs) {
            outputs.push_back(command_t::tid_node_id_t{.tid = std::get<0>(out),
                .node = std::get<1>(out)});
          }

          // create the broadcast command
          return command_t::create_broadcast(cur_cmd++,
                                             command_t::tid_node_id_t{.tid = std::get<0>(cmd.inputs[0]),
                                                 .node = std::get<1>(cmd.inputs[0])},
                                             outputs);
        }
        // is this a simple MOVE?
        else {

          // create the move command
          return command_t::create_move(cur_cmd++,
                                     command_t::tid_node_id_t{.tid = std::get<0>(cmd.inputs[0]),
                                                                 .node = std::get<1>(cmd.inputs[0])},
                                     command_t::tid_node_id_t{.tid = std::get<0>(cmd.outputs[0]),
                                                                  .node = std::get<1>(cmd.outputs[0])});
        }
      }
      case parsed_command_t::op_type_t::DELETE : {

        // make sure there are not outputs
        if(cmd.outputs.size() != 0) {
          throw std::runtime_error("A delete operation must have no outputs.");
        }

        // make there is at least one input
        if(cmd.inputs.size() == 0) {
          throw std::runtime_error("A delete operation must have at least one input.");
        }

        // no need to move here
        std::vector<command_t::tid_node_id_t> inputs;
        for(auto &in : cmd.inputs) {
          inputs.push_back(command_t::tid_node_id_t{.tid = std::get<0>(in),
                                                    .node = std::get<1>(in)});
        }

        // create the delete command
        return command_t::create_delete(cur_cmd++, inputs);
      }
      case parsed_command_t::op_type_t::APPLY : {

        // make sure there is at least one output
        if(cmd.outputs.size() == 0) {
          throw std::runtime_error("An APPLY must have at least one output.");
        }

        // return me that matcher for matrix addition
        auto matcher = _udf_manager.get_matcher_for(cmd.def.ud_name);
        if(matcher == nullptr) {
          throw std::runtime_error("Could not find function " + cmd.def.ud_name +  " in APPLY.");
        }

        // get the ud object
        auto ud = matcher->findMatch(cmd.def.input_types, cmd.def.output_types, cmd.def.is_gpu);
        if(ud == nullptr) {
          throw std::runtime_error("No matching implementation for the function " + cmd.def.ud_name + ".");
        }
        
        std::vector<command_t::tid_node_id_t> inputs;
        for(auto &in : cmd.inputs) {
          inputs.push_back(command_t::tid_node_id_t{.tid = std::get<0>(in),
                                                    .node = std::get<1>(in)});
        }

        std::vector<command_t::tid_node_id_t> outputs;
        for(auto &out : cmd.outputs) {
          outputs.push_back(command_t::tid_node_id_t{.tid = std::get<0>(out),
                                                     .node = std::get<1>(out)});
        }

        return command_t::create_apply(cur_cmd++,
                                       ud->impl_id,
                                       cmd.def.is_gpu,
                                       cmd.parameters,
                                       inputs,
                                       outputs);
      }
      case parsed_command_t::op_type_t::REDUCE : {

        // make sure there is at least one output
        if(cmd.outputs.size() != 1) {
          throw std::runtime_error("An REDUCE must have at exactly one output.");
        }

        // return me that matcher for matrix addition
        auto matcher = _udf_manager.get_matcher_for(cmd.def.ud_name, true, true);
        if(matcher == nullptr) {
          throw std::runtime_error("No matching function " + cmd.def.ud_name +
                                       " that is associative and commutative for a REDUCE.");
        }

        // get the ud object
        auto ud = matcher->findMatch(cmd.def.input_types, cmd.def.output_types, cmd.def.is_gpu);
        if(ud == nullptr) {
          throw std::runtime_error("No matching implementation for the function " + cmd.def.ud_name + ".");
        }

        // make the inputs
        std::vector<command_t::tid_node_id_t> inputs;
        for(auto &in : cmd.inputs) {
          inputs.push_back(command_t::tid_node_id_t{.tid = std::get<0>(in),
                                                    .node = std::get<1>(in)});
        }

        // create the reduce
        return command_t::create_reduce(cur_cmd++,
                                        ud->impl_id,
                                        cmd.def.is_gpu,
                                        cmd.parameters,
                                        inputs,
                                        command_t::tid_node_id_t{.tid = std::get<0>(cmd.outputs[0]),
                                                                 .node = std::get<1>(cmd.outputs[0])});
      }
    }
  }

  // the factory
  tensor_factory_t &_factory;

  // the udf manager
  udf_manager_t &_udf_manager;
};

}