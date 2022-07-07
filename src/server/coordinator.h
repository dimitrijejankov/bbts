#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unistd.h>
#include <dlfcn.h>
#include "../server/coordinator_ops.h"
#include "../server/logger.h"
#include "../commands/command.h"
#include "../commands/reservation_station.h"
#include "../commands/command_runner.h"
#include "../commands/tensor_notifier.h"
#include "../ud_functions/gpu_scheduler.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/udf_manager.h"
#include "../commands/two_layer_compiler.h"
#include "coordinator.h"
#include "static_config.h"

namespace bbts {

class coordinator_t {
public:

  // init the scheduler
  coordinator_t(bbts::communicator_ptr_t _comm,
                bbts::gpu_scheduler_ptr_t _gpu_scheduler,
                bbts::reservation_station_ptr_t _rs,
                bbts::logger_ptr_t _logger,
                storage_ptr_t _storage,
                bbts::command_runner_ptr_t _command_runner,
                bbts::tensor_notifier_ptr_t _tensor_notifier,
                bbts::udf_manager_ptr _udf_manager,
                bbts::tensor_factory_ptr_t _tf);

  // accept a request
  void accept();

  // schedules all the provided commands
  std::tuple<bool, std::string> schedule_commands(const std::vector<command_ptr_t>& cmds);

  // compile the commands
  std::tuple<bool, std::string> compile_commands(float gpu_transfer_cost_per_byte,
                                                 float send_cost_per_byte,
                                                 const std::vector<abstract_command_t>& cmds,
                                                 const std::vector<abstract_ud_spec_t> &funs);

  // run the commands
  std::tuple<bool, std::string> run_commands();

  // set the verbose status
  std::tuple<bool, std::string> set_verbose(bool val);

  // sets the number of threads to use
  std::tuple<bool, std::string> set_num_threads(std::uint32_t set_num_threads);

  // set the maximum storage
  std::tuple<bool, std::string> set_max_storage(size_t set_num_threads);

  // print the info abo
  std::tuple<bool, std::string> print_storage_info();

  // print info about a tensor
  std::tuple<bool, std::string> print_tensor_info(bbts::tid_t id);

  // clears the storage
  std::tuple<bool, std::string> clear();

  // shutdown the cluster
  std::tuple<bool, std::string> shutdown_cluster();

  // load the shared library contained in file_bytes
  std::tuple<bool, std::string> load_shared_library(char* file_bytes, size_t file_size);

  // shutdown the coordinator
  void shutdown();

private:


  
  std::tuple<bool, std::string> _fetch_tensor_info(std::unordered_map<bbts::tid_t, bbts::tensor_meta_t> &meta, 
                                                   std::vector<std::unordered_set<bbts::tid_t>> &locations);

  void _fail();

  void _schedule(coordinator_op_t ops, std::stringstream &ss);

  void _collect(std::tuple<bool, std::string> &out);

  void _load_cmds(const std::vector<command_ptr_t> &cmds,
                  std::stringstream &ss);

  void _run();

  void _clear();

  template<class T = storage_t>
  void _shutdown() {

    // sync
    _comm->barrier();

    // shutdown the gpu scheduler
    _gpu_scheduler->shutdown();

    // shutdown the command runner
    _command_runner->shutdown();

    // shutdown the reservation station
    _rs->shutdown();

    // shutdown the storage
    if constexpr(static_config::enable_storage) {
      std::static_pointer_cast<T>(_storage)->shutdown();
    }

    // shutdown the tensor notifier
    _tensor_notifier->shutdown();

    // mark that the coordinator is down
    _is_down = true;
  }

  void _set_verbose(bool val);

  template<class T = storage_t>
  void _set_max_storage(size_t val) {
    if constexpr(static_config::enable_storage) {
      std::static_pointer_cast<T>(_storage)->set_max_storage(val);
    }
  } 

  void _print_storage(std::stringstream &ss);

  void _print_tensor(tid_t id, std::stringstream &ss);

  // the gpu scheduler
  gpu_scheduler_ptr_t _gpu_scheduler;

  // recieves the bytes of the .so library from another node
  bool _register(coordinator_op_t ops, std::stringstream &ss);

  // register from loaded bytes
  bool _register_from_bytes(char* file_bytes, size_t file_size, std::stringstream &ss);

  // handles a request to fetch the meta data
  void _handle_fetch_meta(std::stringstream &ss);

  // the communicator
  bbts::communicator_ptr_t _comm;

  // this stores all our tensors
  storage_ptr_t _storage;

  // the reservation station
  bbts::reservation_station_ptr_t _rs;

  // the logger
  bbts::logger_ptr_t _logger;

  // shutdown
  std::atomic<bool> _is_down{};

  // runs commands
  bbts::command_runner_ptr_t _command_runner;

  // the notifier
  bbts::tensor_notifier_ptr_t _tensor_notifier;

  // tensor factory
  bbts::tensor_factory_ptr_t _tf;

  // the udf manager
  bbts::udf_manager_ptr _udf_manager;

  // This struct will handle removing temporary files opened by 
  // load_shared_library and hold onto the so_handle. 
  // It isn't strictly necessary because the os will delete files 
  // in /tmp/ periodically. 
  // When barb exits, the shared library will be closed regardless 
  // of calling dlclose.
  struct shared_library_item_t {

    shared_library_item_t(std::string const& filename, 
                          void* so_handle) : filename(filename), 
                                             so_handle(so_handle) {}
    
    ~shared_library_item_t() {

      // we no longer need the temporary file
      unlink(filename.c_str());
    }

    // the file of the loaded library
    std::string filename;

    // the .so file handle
    void* so_handle;

    // the id of the last so library, we use this during the name generation
    static int64_t last_so;
  };

  std::vector<shared_library_item_t> shared_libs;
};

// the pointer
using coordinator_ptr_t = std::shared_ptr<coordinator_t>;

}
