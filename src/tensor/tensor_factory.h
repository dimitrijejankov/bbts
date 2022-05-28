#pragma once

#include <functional>
#include <unordered_map>
#include <memory>
#include "tensor.h"

namespace bbts {

// this class is responsible for registering tensor types
// handling their size, an initializing them
class tensor_factory_t {
public:

  // initializes the tensor factory with built-in types
  tensor_factory_t();

  // registers the tensor format along with the creation functions
  tfid_t register_fmt(const std::string &_fmt_name, const tensor_creation_fs_t& _fmt_funs);

  // returns the impl_id of the tensor format by string if found
  tfid_t get_tensor_ftm(const std::string &_fmt_name);

  // gets the size or at least the upper bound required to store a tensor
  size_t get_tensor_size(const tensor_meta_t& _meta);

  // the tensor to print
  void print_tensor(tensor_t* here, std::stringstream &ss);

  // fills out the meta data of a tensor by preparsing the data for a particular tensor format id
  void deserialize_meta(tensor_meta_t& _meta, tfid_t id, const char *data);

  // deserializes a tensor from the input data
  bbts::tensor_t &deserialize_tensor(tensor_t* here, tfid_t id, const char *data);

  // // create the meta data for the output tensor of the stack operation
  // void create_stack_meta(tensor_meta_t& _meta, std::vector<tensor_meta_t> meta_list_to_stack);

  // // create the output tensor of the stack operation
  // bbts::tensor_t &create_stack_tensor(tensor_t* here, std::vector<tensor_meta_t> meta_list_to_stack);

  // initializes the tensor
  tensor_t& init_tensor(tensor_t* here, const tensor_meta_t& _meta);

private:

  // all the tensor formats registered in the system
  std::unordered_map<std::string, tfid_t> _reg_fmts;

  // maps the fmt impl_id to the functions to create it
  std::vector<tensor_creation_fs_t> _fmt_fs;

};

// just a nicer way to say tensor factory ptr
using tensor_factory_ptr_t = std::shared_ptr<tensor_factory_t>;

}