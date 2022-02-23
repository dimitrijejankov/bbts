#include "ib_communicator.h"

namespace bbts {

using namespace ib;

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

  auto fut = connection.send(
    0,
    com_tag::response_string_tag,
    to_bytes_t(val.data(), val.size()));

  return fut.get();
}

// expect a response string
std::tuple<bool, std::string>
ib_communicator_t::expect_response_string(
  int32_t from_rank)
{
  if(get_rank() != 0) {
    throw std::runtime_error("only node 0 should recv response string");
  }
  if(from_rank == 0) {
    throw std::runtime_error("cannot expect message from self");
  }

  auto [success, bytes] = connection.recv_from(from_rank, com_tag::response_string_tag).get();

  std::string ret;
  if(success) {
    ret = std::string(bytes.ptr.get(), bytes.size);
  }
  return {success, ret};
}

bool ib_communicator_t::send_sync(
  const void *bytes,
  size_t num_bytes,
  node_id_t dest_rank,
  int32_t tag)
{
  return connection.send(
    dest_rank,
    com_tag::free_tag + tag,
    {(void*)bytes, num_bytes}
  ).get();
}

bool ib_communicator_t::recv_sync(
  void *bytes, size_t num_bytes,
  node_id_t from_rank,
  int32_t tag)
{
  return connection.recv_from_with_bytes(
    from_rank,
    com_tag::free_tag + tag,
    {bytes, num_bytes}
  ).get();
}

bool ib_communicator_t::tensors_created_notification(
  node_id_t dest_rank,
  const std::vector<bbts::tid_t> &tensor)
{
  return connection.send(
    dest_rank,
    com_tag::notify_tensor_tag,
    to_bytes_t(tensor.data(), tensor.size())
  ).get();
}

std::tuple<node_id_t, std::vector<bbts::tid_t>>
ib_communicator_t::receive_tensor_created_notification() {
  auto [success, from_rank, bytes] = connection.recv(
    com_tag::notify_tensor_tag
  ).get();

  // just keep doing this method on failure
  if(!success) {
    return receive_tensor_created_notification();
  }

  bbts::tid_t* beg = (bbts::tid_t*)bytes.ptr.get();
  bbts::tid_t* end = (bbts::tid_t*)(bytes.ptr.get() + bytes.size);

  return {from_rank,  std::vector<bbts::tid_t>(beg, end)};
}

bool ib_communicator_t::shutdown_notification_handler() {
  // TODO: support sending to yourself
  std::vector<bbts::tid_t> tensor = { -1 };
  tensors_created_notification(get_rank(), tensor);
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

bool ib_communicator_t::sync_resource_aquisition(
  command_id_t cmd,
  const bbts::command_t::node_list_t &nodes,
  bool my_val)
{
  return true;
}

bool ib_communicator_t::sync_resource_aquisition_p2p(
  command_id_t cmd,
  node_id_t &node,
  bool my_val)
{
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
bool ib_communicator_t::expect_coord_cmds(
  size_t num_cmds,
  std::vector<command_ptr_t> &out)
{
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
