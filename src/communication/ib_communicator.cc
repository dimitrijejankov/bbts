#include "ib_communicator.h"

namespace bbts {

using ib::bytes_t;
using ib::to_bytes_t;
using ib::recv_bytes_t;

enum com_tag {
  response_string // TODO: when adding tags, leave space after response string
};

ib_communicator_t::ib_communicator_t(
  node_config_ptr_t const& _cfg,
  std::string const& dev_name,
  int rank,
  std::vector<std::string> const& ips): connection(dev_name, rank, ips)
{}

ib_communicator_t::~ib_communicator_t() {}

// send a response string
bool ib_communicator_t::send_response_string(const std::string &val) {
  if(get_rank() == 0) {
    throw std::runtime_error("node 0 should not send response string");
  }

  auto fut = connection.send_bytes(
    0,
    com_tag::response_string + get_rank(),
    to_bytes_t(val.c_str(), val.size()));

  return fut.get();
}

// expect a response string
std::tuple<bool, std::string> ib_communicator_t::expect_response_string(int32_t _node) {
  if(get_rank() != 0) {
    throw std::runtime_error("only node 0 should recv response string");
  }
  if(_node == 0) {
    throw std::runtime_error("cannot expect message from self");
  }

  recv_bytes_t bytes = connection.recv_bytes(com_tag::response_string + _node).get();

  std::string ret(bytes.ptr.get(), bytes.size);
  return {true, ret};
}

bool ib_communicator_t::recv_sync(
  void *_bytes, size_t num_bytes,
  node_id_t _node,
  int32_t _tag)
{
  return true;
}

// does the send, method is blocking
bool ib_communicator_t::send_sync(
  const void *_bytes,
  size_t num_bytes,
  node_id_t _node,
  int32_t _tag)
{
  return true;
}

bool ib_communicator_t::tensors_created_notification(node_id_t out_node, const std::vector<bbts::tid_t> &tensor) {
  return true;
}

std::tuple<node_id_t, std::vector<bbts::tid_t>> ib_communicator_t::receive_tensor_created_notification() {
  return {0, {}};
}

bool ib_communicator_t::shutdown_notification_handler() {
  return true;
}

// recieves the request that we got from expect_request_sync
bool ib_communicator_t::receive_request_sync(node_id_t node, int32_t tag, void *bytes, size_t num_bytes) {
  return true;
}

bool ib_communicator_t::op_request(const command_ptr_t &_cmd) {
  return true;
}

bool ib_communicator_t::shutdown_op_request() {
  return true;
}

command_ptr_t ib_communicator_t::expect_op_request() {
  return nullptr;
}

bool ib_communicator_t::sync_resource_aquisition(command_id_t cmd, const bbts::command_t::node_list_t &nodes, bool my_val) {
  return true;
}

bool ib_communicator_t::sync_resource_aquisition_p2p(command_id_t cmd, node_id_t &node, bool my_val) {
  return true;
}

// waits for all the nodes to hit this, should only be used for initialization
void ib_communicator_t::barrier() {
}

bool ib_communicator_t::send_coord_op(const bbts::coordinator_op_t &op) {
  return true;
}

bbts::coordinator_op_t ib_communicator_t::expect_coord_op() {
}

// send the cmds to all nodes
bool ib_communicator_t::send_coord_cmds(const std::vector<command_ptr_t> &cmds) {
  return true;
}

bool ib_communicator_t::send_bytes(char* file, size_t file_size) {
  return true;
}

// expect the a coord op
bool ib_communicator_t::expect_coord_cmds(size_t num_cmds, std::vector<command_ptr_t> &out) {
  return true;
}

bool ib_communicator_t::expect_bytes(size_t num_bytes, std::vector<char> &out) {
  return true;
}

// return the rank
int32_t ib_communicator_t::get_rank() const {
  return connection.get_rank();
}

// return the number of nodes
int32_t ib_communicator_t::get_num_nodes() const {
  return connection.get_num_nodes();
}

}
