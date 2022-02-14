#pragma once

#include <iostream>

#include "../server/node_config.h"
#include "../ud_functions/ud_function.h"
#include "../tensor/tensor.h"
#include "../commands/command.h"
#include "../server/coordinator_ops.h"

#include "infiniband/connection.h"

namespace bbts {

struct ib_communicator_t {
  explicit ib_communicator_t(
    node_config_ptr_t const& _cfg,
    std::string const& dev_name,
    int rank,
    std::vector<std::string> const& ips);

  ~ib_communicator_t();

  // send a response string
  bool send_response_string(const std::string &val);

  // expect a response string
  std::tuple<bool, std::string> expect_response_string(node_id_t _node);

  // recives a blob with the matching tag from a given node, method blocks
  bool recv_sync(void *_bytes, size_t num_bytes, node_id_t _node, int32_t _tag);

  // does the send, method is blocking
  bool send_sync(const void *_bytes, size_t num_bytes, node_id_t _node, int32_t _tag);

  // wait for async request
  //bool wait_async(async_request_t &_request);

  // notify a node that tensors were created
  bool tensors_created_notification(
    node_id_t out_node, const std::vector<bbts::tid_t> &tensor);

  // wait to receive a notification
  std::tuple<node_id_t, std::vector<bbts::tid_t>> receive_tensor_created_notification();

  // shutdown the notification handler
  bool shutdown_notification_handler();

  // recieves the request that we got from expect_request_sync
  bool receive_request_sync(node_id_t node, int32_t tag, void *bytes, size_t num_bytes);

  // initiates the operation on all the specified nodes
  bool op_request(const command_ptr_t &_cmd);

  // waits to recieve an operation
  command_ptr_t expect_op_request();

  // this sends a shutdown command to the thread that is calling @see expect_op_request
  bool shutdown_op_request();

  // send the coord op to all nodes
  bool send_coord_op(const bbts::coordinator_op_t &op);

  // expect the a coord op
  bbts::coordinator_op_t expect_coord_op();

  // send the cmds to all nodes
  bool send_coord_cmds(const std::vector<command_ptr_t> &cmds);

  // send a bunch of bytes to all nodes
  bool send_bytes(char* file, size_t file_size);

  // expect the a coord op
  bool expect_coord_cmds(size_t num_cmds, std::vector<command_ptr_t> &out);

  // sync the resource aquisition
  bool sync_resource_aquisition(
    command_id_t cmd,
    const bbts::command_t::node_list_t &nodes,
    bool my_val);

  // sync the resource aquisition between two nodes
  bool sync_resource_aquisition_p2p(command_id_t cmd, node_id_t &node, bool my_val);

  // recieved the tensors size
  std::tuple<uint64_t, bool> recv_tensor_size(node_id_t node, int32_t tag);

  // send the tensor size
  bool send_tensor_size(node_id_t node, int32_t tag, uint64_t val);

  bool expect_bytes(size_t num_bytes, std::vector<char> &out);

  // waits for all the nodes to hit this, should only be used for initialization
  void barrier();

  // return the rank
  int32_t get_rank() const;

  // return the number of nodes
  int32_t get_num_nodes() const;
private:
  ib::connection_t connection;
};

} // bbts
