#include "builtin_formats.h"
#include "tensor.h"
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>

namespace bbts {

tensor_creation_fs_t bbts::dense_tensor_t::get_creation_fs() {

  // return the init function
  auto init = [](void *here, const tensor_meta_t &_meta) -> tensor_t & {
    auto &t = *(dense_tensor_t *) here;
    auto &m = *(dense_tensor_meta_t * ) & _meta;
    t.meta() = m;
    return t;
  };

  // return the size
  auto size = [](const tensor_meta_t &_meta) {
    auto &m = *(dense_tensor_meta_t *) &_meta;
    return sizeof(tensor_meta_t) + m.m().num_cols * m.m().num_rows * sizeof(float);
  };

  auto pnt = [](const void *here, std::stringstream &ss) {
    
    // get the tensor
    auto &t = *(dense_tensor_t *) here;

    // extract the info
    auto num_rows = t.meta().m().num_rows;
    auto num_cols = t.meta().m().num_cols;
    auto data = t.data();

    // print the tensor
    for(int i = 0; i < num_rows; i++) {
      ss << "[";
      for(int j = 0; j < num_cols; j++) {
        ss << data[i * num_cols + j] << ((j == num_cols - 1) ? "" : ",");
      }
      ss << "]\n";
    }

  };

  auto deserialize_meta = [](tensor_meta_t& _meta, tfid_t id, const char *data) {
    auto &m = *(dense_tensor_meta_t *) &_meta;
    m.fmt_id = id;

    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    m.m().num_rows = std::atoi(num_rows.c_str());
    m.m().num_cols = std::atoi(num_columns.c_str());

  };

  auto deserialize_tensor = [](tensor_t* here, tfid_t id, const char *data) -> tensor_t& {
    
    auto &a = here->as<dense_tensor_t>();
    // set meta data
    // tfid
    a.meta().fmt_id = id;

    // number of rows and columns
    auto s = std::string(data);
    std::string delimiter = "|";
    size_t pos = s.find(delimiter);
    std::string num_rows = s.substr(0, s.find(delimiter));
    s.erase(0, pos + delimiter.length());
    std::string num_columns = s.substr(0, s.find(delimiter));

    a.meta().m().num_rows = std::atoi(num_rows.c_str());
    a.meta().m().num_cols = std::atoi(num_columns.c_str());

    // put actual data inside tensor
    s.erase(0, pos + delimiter.length());
    std::string data_delimiter = " ";
    size_t data_pos = s.find(delimiter);

    for (auto row = 0; row < a.meta().m().num_rows; ++row) {
      for (auto col = 0; col < a.meta().m().num_cols; ++col) {
        data_pos = s.find(data_delimiter);
        std::string my_data = s.substr(0, data_pos);
        s.erase(0, data_pos + data_delimiter.length());
        auto temp = std::atof(my_data.c_str());
        a.data()[row * a.meta().m().num_cols + col] = temp;
      }
    }
    
    return a;
  };

  // auto create_stack_meta = [](tensor_meta_t& _meta, std::vector<tensor_meta_t> meta_list_to_stack){
  //   auto &m = *(dense_tensor_meta_t *) &_meta;
  //   // We want to output a dense tensor so currently to fmt id = 0 
  //   // NOTICE: MAY CAUSE REFACTORING PROBLEMS SINCE IT IS HARDCODED
  //   m.fmt_id = 0;

  //   int num_rows = 0;
  //   int num_cols = (*(dense_tensor_meta_t *) &meta_list_to_stack[0]).m().num_cols;

  //   for (tensor_meta_t single_meta: meta_list_to_stack){
  //     auto &single_dense_meta = *(dense_tensor_meta_t *) &single_meta;
  //     num_rows = num_rows + single_dense_meta.m().num_rows;
  //     // error check for cols
  //     if (num_cols != single_dense_meta.m().num_cols){
  //       // TODO: we need to find a way to stop the system, maybe use a boolean for the return type?
  //       std::cout << "DIMENSION (NUM OF COLUMNS) INCOMPATIBLE\n";
  //     }
  //   }


  //   m.m().num_rows = num_rows;
  //   m.m().num_cols = num_cols;

  
  // };

  // auto create_stack_tensor = [](tensor_t* here, std::vector<tensor_meta_t> meta_list_to_stack){
  //   auto &a = here->as<dense_tensor_t>();
  //   // set meta data
  //   // We want to output a dense tensor so currently to fmt id = 0 
  //   // NOTICE: MAY CAUSE REFACTORING PROBLEMS SINCE IT IS HARDCODED
  //   a.meta().fmt_id = 0;

  //   int num_rows = 0;
  //   int num_cols = (*(dense_tensor_meta_t *) &meta_list_to_stack[0]).m().num_cols;

  //   for (tensor_meta_t single_meta: meta_list_to_stack){
  //     auto &single_dense_meta = *(dense_tensor_meta_t *) &single_meta;
  //     num_rows = num_rows + single_dense_meta.m().num_rows;
  //     // error check for cols
  //     if (num_cols != single_dense_meta.m().num_cols){
  //       // TODO: we need to find a way to stop the system, maybe use a boolean for the return type?
  //       std::cout << "DIMENSION (NUM OF COLUMNS) INCOMPATIBLE\n";
  //     }
  //     // TODO: Load the tensor data here
  //   }


  //   a.meta().m().num_rows = num_rows;
  //   a.meta().m(),num_cols = num_cols;

  //   return a;

  // };


    return tensor_creation_fs_t{.get_size = size, 
                              .init_tensor = init, 
                              .print = pnt, .deserialize_meta = deserialize_meta, 
                              .deserialize_tensor = deserialize_tensor};



}

}