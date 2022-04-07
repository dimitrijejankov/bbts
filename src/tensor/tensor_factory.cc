#include "tensor_factory.h"
#include "builtin_formats.h"
#include <sstream>

bbts::tfid_t bbts::tensor_factory_t::register_fmt(const std::string &_fmt_name,
                                                  const bbts::tensor_creation_fs_t &_fmt_funs) {

  // check if the format exits
  auto it = _reg_fmts.find(_fmt_name);
  if(it == _reg_fmts.end()) {
    _reg_fmts[_fmt_name] = _fmt_fs.size();
    _fmt_fs.emplace_back(_fmt_funs);
  }

  // we can not register twice the same format
  return -1;
}

bbts::tensor_t &bbts::tensor_factory_t::init_tensor(tensor_t *here, const bbts::tensor_meta_t &_meta) {

  // find the function to initialize the tensor, run it and set the right fmt_id
  if(_meta.fmt_id < _fmt_fs.size()) {
    bbts::tensor_t &out = _fmt_fs[_meta.fmt_id].init_tensor(here, _meta);
    out.as<bbts::tensor_meta_t>().fmt_id = _meta.fmt_id;
    return out;
  }

  // check if the format exits
  throw std::runtime_error("Requested init_tensor for a format " + std::to_string(_meta.fmt_id) + " not registered with the system.");
}

size_t bbts::tensor_factory_t::get_tensor_size(const bbts::tensor_meta_t &_meta) {

  // find the function and return the size
  if(_meta.fmt_id < _fmt_fs.size()) {
    return _fmt_fs[_meta.fmt_id].get_size(_meta);
  }

  // check if the format exits
  throw std::runtime_error("Requested get_tensor_size for a format " + std::to_string(_meta.fmt_id) + " not registered with the system.");
}

void bbts::tensor_factory_t::print_tensor(tensor_t* here, std::stringstream &ss) {

  // find the function to initialize the tensor, run it and set the right fmt_id
  if(here->as<bbts::tensor_meta_t>().fmt_id < _fmt_fs.size()) {
    _fmt_fs[here->as<bbts::tensor_meta_t>().fmt_id].print(here, ss);
    return;
  }

  throw std::runtime_error("Requested init_tensor for a format " + std::to_string(here->as<bbts::tensor_meta_t>().fmt_id) + " not registered with the system.");
}

bbts::tensor_factory_t::tensor_factory_t() {

  // register the dense format
  register_fmt("dense", dense_tensor_t::get_creation_fs());
}

bbts::tfid_t bbts::tensor_factory_t::get_tensor_ftm(const std::string &_fmt_name) {

  // check if the format exits
  auto it = _reg_fmts.find(_fmt_name);
  if(it != _reg_fmts.end()) {
    return it->second;
  }
  return -1;
}
