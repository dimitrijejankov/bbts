#pragma once

#include <cassert>
#include <fstream>
#include "../server/node_config.h"
#include "../tensor/tensor.h"
#include "../utils/raw_vector.h"
#include "command_utils.h"

namespace bbts {

// this represents a command that was parsed it is not yet translated
struct parsed_command_t {

  // the definition of a ud function
  struct ud_def_t {

    // the name of the user define function
    std::string ud_name;

    // the input types
    std::vector<std::string> input_types;

    // the output types
    std::vector<std::string> output_types;

    // are we requesting a gpu based ud function
    bool is_gpu = false;
  };

  // the type of the operation
  enum class op_type_t : int32_t {

    APPLY = 0,
    REDUCE = 1,
    MOVE = 2,
    DELETE = 3
  };

  // the type of ethe command
  op_type_t type;

  // the definition of the ud function
  ud_def_t def;

  // the inputs of this op
  std::vector<std::tuple<tid_t, node_id_t>> inputs;

  // the outputs of this op
  std::vector<std::tuple<tid_t, node_id_t>> outputs;

  // the parameters of the ud function
  std::vector<command_param_t> parameters;

  void print_tensor_list(const std::vector<std::tuple<tid_t, node_id_t>> &list, std::ostream &out) const {

    out << "[";
    for(size_t i = 0; i < list.size(); ++i) {
      if(i == list.size() - 1) {
        out << "(" << std::get<0>(list[i]) << "," << std::get<1>(list[i]) << ")";
      }
      else {
        out << "(" << std::get<0>(list[i]) << "," << std::get<1>(list[i]) << "),";
      }
    }
    out << "]";
  }

  void print_string_list(const std::vector<std::string> &list, std::ostream &out) const {

    out << "[";
    for(size_t i = 0; i < list.size(); ++i) {
      if(i == list.size() - 1) {
        out << "\"" << list[i] << "\"";
      }
      else {
        out << "\"" << list[i] << "\",";
      }
    }
    out << "]";
  }

  void print(std::ostream &out) const {

    switch(type) {
      case op_type_t::APPLY : {
        out << "APPLY (";
        out << "\"" << def.ud_name << "\",";
        print_string_list(def.input_types, out);
        out << ",";
        print_string_list(def.output_types, out);
        out << ",";
        out << def.is_gpu;
        out << ",";
        print_tensor_list(inputs, out);
        out << ',';
        print_tensor_list(outputs, out);
        out << ")\n";

        break;
      }
      case op_type_t::REDUCE : {
        out << "REDUCE (";
        out << "\"" << def.ud_name << "\",";
        print_string_list(def.input_types, out);
        out << ",";
        print_string_list(def.output_types, out);
        out << ",";
        out << def.is_gpu;
        out << ",";
        print_tensor_list(inputs, out);
        out << ',';
        print_tensor_list(outputs, out);
        out << ")\n";

        break;
      }
      case op_type_t::MOVE : {
        out << "MOVE (";
        print_tensor_list(inputs, out);
        out << ',';
        print_tensor_list(outputs, out);
        out << ")\n";
        break;
      }
      case op_type_t::DELETE : {
        out << "DELETE (";
        print_tensor_list(inputs, out);
        out << ")\n";
        break;
      }
    }

  }

};

// TODO - the deserialization can be optimized if necessary
struct parsed_command_list_t {

  void add_move(const std::tuple<tid_t, node_id_t> &in,
                const std::vector<std::tuple<tid_t, node_id_t>> &out) {

    _commands.push_back(parsed_command_t{.type = parsed_command_t::op_type_t::MOVE,
                                         .def = {},
                                         .inputs = { in },
                                         .outputs = out});
  }

  void add_apply(const std::string &fn,
                 const std::vector<std::string> &in_types,
                 const std::vector<std::string> &out_types,
                 bool is_gpu,
                 const std::vector<std::tuple<tid_t, node_id_t>> &in,
                 const std::vector<std::tuple<tid_t, node_id_t>> &out,
                 const std::vector<command_param_t> &params) {

    _commands.push_back(parsed_command_t{.type = parsed_command_t::op_type_t::APPLY,
                                         .def = {.ud_name = fn,
                                                 .input_types = in_types,
                                                 .output_types = out_types,
                                                 .is_gpu = is_gpu},
                                         .inputs = in,
                                         .outputs = out,
                                         .parameters = params});
  }

  void add_reduce(const std::string &fn,
                  const std::vector<std::string> &in_types,
                  const std::vector<std::string> &out_types,
                  bool is_gpu,
                  const std::vector<std::tuple<tid_t, node_id_t>> &in,
                  const std::tuple<tid_t, node_id_t> &out,
                  const std::vector<command_param_t> &params) {

    _commands.push_back(parsed_command_t{.type = parsed_command_t::op_type_t::REDUCE,
                                         .def = {.ud_name = fn,
                                                 .input_types = in_types,
                                                 .output_types = out_types,
                                                 .is_gpu = is_gpu},
                                         .inputs = in,
                                         .outputs = { out },
                                         .parameters = params});
  }

  void add_delete(const std::vector<std::tuple<tid_t, node_id_t>> &in) {

    _commands.push_back(parsed_command_t{.type = parsed_command_t::op_type_t::DELETE,
                                         .def = {},
                                         .inputs = in,
                                         .outputs = {}});
  }

  bool deserialize(const std::string &file_name) {

    // try to open the file
    std::ifstream file;
    file.open(file_name, std::ios::in | std::ios::binary);

    // did we manage to open the file?
    if (!file.is_open()) {
      return false;
    }

    // create the index and the commands
    _commands.clear();

    // read the string table size
    auto num_commands = deserialize_val<uint64_t>(file);
    for(size_t idx = 0; idx < num_commands; ++idx) {
      deserialize_command(file);
    }

    return true;
  }

  bool serialize(const std::string &file_name) {

    // try to open the file
    std::ofstream file;
    file.open(file_name, std::ios::out | std::ios::trunc | std::ios::binary);

    // did we manage to open the file?
    if (!file.is_open()) {
      return false;
    }

    // write the number of commands
    serialize_val(file, _commands.size());
    for(auto &cmd : _commands) {
      serialize_cmd(cmd, file);
    }

    return true;
  }

  const parsed_command_t& operator[](size_t idx) const { return _commands[idx]; };

  size_t get_num_commands() const {
    return _commands.size();
  }

  void print(std::ostream &out) const {

    // print each command
    for(auto &cmd : _commands) {
      cmd.print(out);
    }
  }

 private:

  void serialize_cmd(const parsed_command_t &cmd, std::ofstream &file) {

    // write the command type
    serialize_val(file, cmd.type);
    
    // write the ud function name
    serialize_string(file, cmd.def.ud_name);
    serialize_val(file, cmd.def.is_gpu);

    // write the number of input and output types
    serialize_strings(file, cmd.def.input_types);
    serialize_strings(file, cmd.def.output_types);

    // serialize the input and output array
    serialize_array(file, cmd.inputs);
    serialize_array(file, cmd.outputs);
    serialize_array(file, cmd.parameters);
  }

  void serialize_strings(std::ofstream &file, const std::vector<std::string> &strs) {

    // serialize the strings
    serialize_val(file, strs.size());

    // write all the strings
    for(auto &str : strs) {
      serialize_string(file, str);
    }
  }

  void serialize_string(std::ofstream &file, const std::string &str) {

    // serialize the string size
    serialize_val(file, str.size());

    // serialize the string data
    if(!str.empty()) {
      file.write(str.data(), str.size());
    }
  }

  template<class T>
  void serialize_array(std::ofstream &file, const std::vector<T> tmp) {

    // serialize the array
    serialize_val(file, tmp.size());
    file.write((char*) tmp.data(), tmp.size() * sizeof(T));
  }

  template<class T>
  void serialize_val(std::ofstream &file, T tmp) {
    file.write((char*) &tmp, sizeof(T));
  }

  void deserialize_command(std::ifstream &file) {

    // create a new command
    _commands.push_back({});
    auto &cmd = _commands.back();

    // write the command type
    cmd.type = deserialize_val<decltype(cmd.type)>(file);

    // write the ud function name
    cmd.def.ud_name = deserialize_string(file);
    cmd.def.is_gpu = deserialize_val<decltype(cmd.def.is_gpu)>(file);

    // write the number of input and output types
    cmd.def.input_types = deserialize_strings(file);
    cmd.def.output_types = deserialize_strings(file);

    // serialize the input and output array
    cmd.inputs = deserialize_array<std::tuple<tid_t, node_id_t>>(file);
    cmd.outputs = deserialize_array<std::tuple<tid_t, node_id_t>>(file);
    cmd.parameters = deserialize_array<command_param_t>(file);
  }

  std::vector<std::string> deserialize_strings(std::ifstream &file) {

    // get the number of strings
    auto num_strings = deserialize_val<uint64_t>(file);;
    std::vector<std::string> out(num_strings);

    // deserialize all the strings
    for(auto &s : out) {
      s = deserialize_string(file);
    }

    return std::move(out);
  }

  std::string deserialize_string(std::ifstream &file) {

    // get the number of character in the string
    std::string tmp;
    auto numChars = deserialize_val<uint64_t>(file);

    // set the data
    if(numChars != 0) {
      tmp.resize(numChars);
      file.read(tmp.data(), numChars);
    }

    return tmp;
  }

  template<class T>
  std::vector<T> deserialize_array(std::ifstream &file) {

    // get the number of elements
    auto num_elements = deserialize_val<uint64_t>(file);;
    std::vector<T> out(num_elements);

    // deserialize all the elements
    for(auto &s : out) {
      s = deserialize_val<T>(file);
    }

    return std::move(out);
  }

  template<class T>
  T deserialize_val(std::ifstream &file) {
    T tmp;
    file.read((char*) &tmp, sizeof(T));
    return tmp;
  }

  // the commands
  std::vector<parsed_command_t> _commands;
};

}