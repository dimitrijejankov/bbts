#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <sys/sysinfo.h>

#include "logger.h"
#include "node_config.h"
#include "static_config.h"
#include "coordinator.h"
#include "../ud_functions/udf_manager.h"
#include "../commands/command_runner.h"
#include "../commands/tensor_notifier.h"
#include "../commands/parsed_command.h"
#include "../ud_functions/gpu_scheduler.h"


namespace bbts {

class node_t {
public:

  // the event that happened
  enum class event_t {

    COMMAND_QUEUED,
    COMMAND_SCHEDULED,
    COMMAND_RETIRED,
    TENSOR_DELETED,
    TENSOR_CREATED
  };

  // creates the node
  explicit node_t(bbts::node_config_ptr_t config) {

    // move the config
    _config = std::move(config);
  }

  // initialize stuff like communication
  void init();

  // starts handling the commands
  void run();

  // return the number of nodes
  [[nodiscard]] size_t get_num_nodes() const;

  // return the rank of the node
  [[nodiscard]] size_t get_rank() const;

  // print the cluster info
  void print_cluster_info(std::ostream& out);

  // load all commands
  std::tuple<bool, std::string> load_commands(const std::vector<command_ptr_t>& cmds);

  // load all the parsed commands
  std::tuple<bool, std::string> load_commands(const bbts::parsed_command_list_t &cmds);

  // compile the commands
  std::tuple<bool, std::string> compile_commands(const std::string &file_path);

  // load a shared library
  std::tuple<bool, std::string> load_shared_library(char* file_bytes, size_t file_size);

  // run all the scheduled commands
  std::tuple<bool, std::string> run_commands();

  // set the verbose
  std::tuple<bool, std::string> set_verbose(bool val);

  // set the number of threads
  std::tuple<bool, std::string> set_num_threads(std::uint32_t set_num_threads);

  // set the storage
  std::tuple<bool, std::string> set_max_storage(size_t set_num_threads);

  // print the info about the storage
  std::tuple<bool, std::string> print_storage_info();

  // prints information about the tensor
  std::tuple<bool, std::string> print_tensor_info(tid_t id);

  // resets the cluster
  std::tuple<bool, std::string> clear();

  // shutdown cluster
  std::tuple<bool, std::string> shutdown_cluster();

  // sync all the nodes to know that they have all execute up to this point
  void sync();

  // add the hook
  template<event_t event, class fn>
  void add_hook(fn fun) {

    // add the relevant hooks
    if constexpr (event == event_t::COMMAND_QUEUED) {
      _res_station->add_queued_hook(fun);
    }
    else if constexpr (event == event_t::COMMAND_SCHEDULED) {
      _res_station->add_scheduled_hook(fun);
    }
    else if constexpr(event == event_t::COMMAND_RETIRED) {
      _res_station->add_retired_hook(fun);
    }
    else if constexpr (event == event_t::TENSOR_CREATED) {
      _storage->add_created_hook(fun);
    }
    else if constexpr (event == event_t::TENSOR_DELETED) {
      _storage->add_deleted_hook(fun);
    }
  }

// protected: for now everything is public

  // the reservation station needs a deleter thread
  std::thread create_deleter_thread();

  std::thread create_move_processing_thread();
  std::thread create_apply_processing_thread();
  std::thread create_reduce_processing_thread();

  std::thread expect_remote_command();

  std::thread remote_tensor_notification_sender(node_id_t out_node);

  std::thread tensor_notifier();

  std::thread create_coordinator_thread();

  // the configuration of the node
  node_config_ptr_t _config;

  // the logger
  logger_ptr_t _logger;

  // the communicator, this is doing all the sending
  communicator_ptr_t _comm;

  // the reservation station
  reservation_station_ptr_t _res_station;

  // this is responsible for forwarding all the commands to the right node and receiving commands
  coordinator_ptr_t _coordinator;

  // this initializes the tensors
  tensor_factory_ptr_t _factory;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the udf manager
  udf_manager_ptr _udf_manager;

  // runs commands
  command_runner_ptr_t _command_runner;

  // the notifier
  tensor_notifier_ptr_t _tensor_notifier;

  // the gpu scheduler
  gpu_scheduler_ptr_t _gpu_scheduler;
};



}
