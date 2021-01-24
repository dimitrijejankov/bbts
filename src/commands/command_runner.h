#pragma once

#include <memory>
#include "reservation_station.h"
#include "../ud_functions/udf_manager.h"
#include "../communication/communicator.h"

namespace bbts {

class command_runner_t {
public:

  command_runner_t(storage_ptr_t ts,
                   tensor_factory_ptr_t tf,
                   udf_manager_ptr udm,
                   reservation_station_ptr_t rs,
                   communicator_ptr_t comm);

  // runs local command
  void local_command_runner();

  // handles the incoming request for remote commands
  void remote_command_handler();

  // run the deleter, responsible to remove all the tensors we don't need anymore...
  void run_deleter();

private:

  // the storage
  bbts::storage_ptr_t _ts;

  // tensor factory
  bbts::tensor_factory_ptr_t _tf;

  // udf manager
  bbts::udf_manager_ptr _udm;

  // reservation station
  bbts::reservation_station_ptr_t _rs;

  // the communicator
  bbts::communicator_ptr_t _comm;
};

using command_runner_ptr_t = std::shared_ptr<command_runner_t>;

}