#include "reduce_notifier.h"

bbts::reduce_notifier_t::reduce_notifier_t(bbts::communicator_ptr_t comm,
                                           bbts::reservation_station_ptr_t rs) : _comm(std::move(comm)),
                                                                                 _rs(std::move(rs)) {}

void bbts::reduce_notifier_t::run_notification_sender_for_node(bbts::node_id_t out_node) {

  while (true) {

    // get tensors to notify the other node
    bool is_done;
    auto reduces = _rs->reduce_to_notify_node(out_node, is_done);
    
    // if it is node break out
    if (is_done) {
      break;
    }

    // add the remote commands
    if(!_comm->tensors_created_notification(out_node, reduces)) {
      throw std::runtime_error("Could not set the tensor notification");
    }
  }
}

void bbts::reduce_notifier_t::run_notification_handler() {

  while (true) {

    // wait for the command
    auto [node, reduces] = _comm->receive_reduce_finished_notification();

    // check if we are done...
    if (reduces[0] == -1) {
      break;
    }

    // notify that the tensors became available
    _rs->notify_ready_reduce(node, reduces);
  }
}

void bbts::reduce_notifier_t::shutdown() {
  if(!_comm->shutdown_notification_handler()){
    throw std::runtime_error("Failed to shutdown!");
  }
}