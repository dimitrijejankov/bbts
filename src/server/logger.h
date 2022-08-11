#pragma once

#include "node_config.h"
#include "../utils/terminal_color.h"
#include <iostream>
#include <mutex>

namespace bbts {

class logger_t {
public:

  logger_t(const node_config_ptr_t &config) : _config(config) {}

  void message(const std::string &text) {

    if(_config->verbose) {
      std::cout << text << std::flush;
    }
  }

  void error(const std::string &text) {

    if(_config->verbose) {
      std::cout << bbts::red << text << bbts::reset << std::flush;
    }
  }

  // print out a warning
  void warn(const std::string &text) {

    if(_config->verbose) {
      std::cout << bbts::yellow << text << bbts::reset << std::flush;
    }
  }

  void set_enabled(bool val) {
    std::unique_lock<std::mutex> lck(m);
    _config->verbose = val;
  }

private:

  // the mutex to lock the verbose flag
  std::mutex m;

  // the config of the node
  node_config_ptr_t _config;
};

using logger_ptr_t = std::shared_ptr<logger_t>;

}