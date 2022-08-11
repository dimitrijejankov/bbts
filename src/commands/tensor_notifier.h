#pragma once

#include "../server/node_config.h"
#include "reservation_station.h"
#include "../communication/communicator.h"

namespace bbts {

class tensor_notifier_t {

public:

  // creates the notifier
  tensor_notifier_t(bbts::communicator_ptr_t comm,
                    bbts::reservation_station_ptr_t rs);

  // sends all the notifications from the reservation station to the out_node
  void run_notification_sender_for_node(bbts::node_id_t out_node);

  // handles notifications sent to this node
  void run_notification_handler();

  // shuts down the notification handler only as the reservation station does that for the sender
  void shutdown();

private:

  // the communicator used to send and receive the notifications
  bbts::communicator_ptr_t _comm;

  // the reservation station
  reservation_station_ptr_t _rs;
};

// nice way to say shared pointer
using tensor_notifier_ptr_t = std::shared_ptr<tensor_notifier_t>;

}