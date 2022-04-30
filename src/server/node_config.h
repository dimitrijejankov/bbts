#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <sys/sysinfo.h>

namespace bbts {

// the type that identifies a node
using node_id_t = int32_t;

// this structure information on how the node is configured
struct node_config_t {

  node_config_t(int32_t argc, char **argv, bool is_dev = false) {

    // mark this is a dev cusluter if set
    is_dev_cluster = is_dev;

    // check if we have something to parse
    if(argc == 0) { return; }

    // parse each flag
    for(int32_t idx = 0; idx < argc; ++idx){

      // check each argument
      if(std::string(argv[idx]) == "--dev") {
        is_dev_cluster = true;
      }
    }
  }

  // the number of threads we are running
  size_t num_threads = std::thread::hardware_concurrency() / 2;

  // the total available ram memory
  size_t get_total_ram() const {
    struct sysinfo info;
    if (sysinfo(&info) < 0) {
      throw std::runtime_error("Failed to initialize the node configuration : Could not get the free RAM\n");
    }
    return info.freeram;
  }
  
  // reserved ram memory
  size_t reserved_ram = 10lu * 1024lu * 1024lu * 1024lu;

  // returns the free memory
  size_t get_free_ram() const {
    struct sysinfo info;
    if (sysinfo(&info) < 0) {
      throw std::runtime_error("Failed to initialize the node configuration : Could not get the free RAM\n");
    }
    return info.freeram;
  }

  // is this a local cluster that we use for testing
  bool is_dev_cluster = false;

  // dev cluser max ram only used if is_dev_cluster is set to true
  size_t dev_cluster_ram = 10lu * 1024lu * 1024lu * 1024lu;

  // should we print out everything?
  bool verbose = false;

  // the cost to transfter bytes to/from the GPU per byte
  float gpu_transfer_cost_per_byte = 1.0f;

  // the cost to send bytes per byte
  float send_cost_per_byte = 1.16415322E-9;

  // the root of the web stuff (html, css and js files)
  std::string web_root = "./web";

  // the directory where all the logs are stored
  std::string log_dir = "./log";

  // web interface port
  int32_t web_port = 8081;
};

// a nice way to reference the configuration ptr
using node_config_ptr_t = std::shared_ptr<node_config_t>;

}