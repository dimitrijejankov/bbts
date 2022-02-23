#include "ib_communicator.h"

namespace bbts {

using namespace ib;

bool wait_all_bools(std::vector<std::future<bool>>& futs) {
  bool success = true;
  for(auto& fut: futs) {
    success = success & fut.get();
  }
  return success;
}

ib_communicator_t::ib_communicator_t(
  node_config_ptr_t const& _cfg,
  std::string const& dev_name,
  int rank,
  std::vector<std::string> const& ips):
    connection(dev_name, rank, com_tag::free_tag, ips)
    // ^ pin all tags before the free tag
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
  std::vector<bbts::tid_t> tensor = { -1 };
  tensors_created_notification(get_rank(), tensor);
}

bool ib_communicator_t::op_request(const command_ptr_t &_cmd) {
  // find all the nodes referenced in the input
  std::vector<node_id_t> to_send_to;
  auto nodes = _cmd->get_nodes();
  for(int node : nodes) {
    if(node != get_rank()) {
      to_send_to.push_back(node);
    }
  }

  std::vector<std::future<bool>> futs;
  for(auto dest_rank : to_send_to) {
    futs.push_back(
      connection.send(
        dest_rank,
        com_tag::send_cmd_tag,
        {_cmd.get(), _cmd->num_bytes()}));
  }

  // wait for all the sends to finish
  return wait_all_bools(futs);
}

bool ib_communicator_t::shutdown_op_request() {
  // create a shutdown command to send to the remote handler
  command_ptr_t cmd = command_t::create_shutdown(get_rank());

  return connection.send(
            get_rank(),
            com_tag::send_cmd_tag,
            {cmd.get(), cmd->num_bytes()}).get();
}

command_ptr_t ib_communicator_t::expect_op_request() {
  // recv a message from anywhere
  auto [success, from_rank, own_bytes] = connection.recv(com_tag::send_cmd_tag).get();

  // check for errors
  if(!success) {
    return nullptr;
  }

  // cast it to the command
  auto p_rel = own_bytes.ptr.release();
  auto p_cmd = (bbts::command_t *)(p_rel);
  auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

  // move the command
  return std::move(d);
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
  // TODO: Implement an bbts::ib::connection_t::barrier
  // TODO: Guessing this will do as a "barrier"
  char c;
  bytes_t bs = to_bytes_t(&c, 1);

  // send bs to every other node, recv from every other node
  std::vector<std::future<bool                    >> send_futs;
  std::vector<std::future<tuple<bool, own_bytes_t>>> recv_futs;
  for(node_id_t node = 0; node < get_num_nodes(); ++node) {
    if(node == get_rank()) {
      continue;
    }
    send_futs.push_back(connection.send(
      node,
      com_tag::barrier_tag,
      bs));
    recv_futs.push_back(connection.recv_from(
      node,
      com_tag::barrier_tag));
  }

  for(int i = 0; i != send_futs.size(); ++i) {
    bool success_send = send_futs[i].get();
    auto [success_recv, _] = recv_futs[i].get();
    if(!success_send || !success_recv) {
      throw std::runtime_error("ib_communicator barrier failed");
    }
  }
}

bool ib_communicator_t::send_coord_op(const bbts::coordinator_op_t &op) {
  // send the op to all nodes except node zero
  std::vector<std::future<bool>> futs;
  futs.reserve(get_num_nodes());
  for(node_id_t node = 1; node < get_num_nodes(); ++node) {
    futs.push_back(
      connection.send(
        node,
        com_tag::coordinator_tag,
        to_bytes_t(&op, 1)));
  }

  // wait for all the requests to finish
  return wait_all_bools(futs);
}

bbts::coordinator_op_t ib_communicator_t::expect_coord_op() {
  bbts::coordinator_op_t op{};
  auto [success, from_rank] = connection.recv_with_bytes(
    com_tag::coordinator_tag,
    to_bytes_t(&op, 1)).get();

  // check for errors
  if(!success) {
    op._type = coordinator_op_types_t::FAIL;
    return op;
  }

  return op;
}

// send the cmds to all nodes
bool ib_communicator_t::send_coord_cmds(const std::vector<command_ptr_t> &cmds) {
  // send all the commands to all nodes except node 0
  for(auto &cmd : cmds) {
    std::vector<std::future<bool>> futs;
    futs.reserve(get_num_nodes());

    for(node_id_t node = 1; node < get_num_nodes(); ++node) {
      futs.push_back(
        connection.send(
          node,
          com_tag::coordinator_bcast_cmd_tag,
          {cmd.get(), cmd->num_bytes()}));
    }

    if(!wait_all_bools(futs)) {
      return false;
    }
  }

  return true;
}

// expect the a coord op
bool ib_communicator_t::expect_coord_cmds(
  size_t num_cmds,
  std::vector<command_ptr_t> &out)
{
  // recv one command at a time and if any fails, stop
  out.reserve(num_cmds);
  for(size_t i = 0; i < num_cmds; ++i) {
    auto [success, _, own_bytes] = connection.recv(com_tag::coordinator_bcast_cmd_tag).get();
    if(success){
      // cast it to the command
      auto p_rel = own_bytes.ptr.release();
      auto p_cmd = (bbts::command_t *)(p_rel);
      auto d = std::unique_ptr<bbts::command_t, command_deleter_t>(p_cmd);

      out.push_back(std::move(d));
    } else {
      return false;
    }
  }
  return true;
}

bool ib_communicator_t::send_bytes(char* file, size_t file_size) {
  // send it everywhere except the root node
  std::vector<std::future<bool>> futs;
  futs.reserve(get_num_nodes());

  for(node_id_t node = 1; node < get_num_nodes(); ++node) {
    futs.push_back(
      connection.send(
        node,
        coordinator_bcast_bytes,
        {(void*)file, file_size}));
  }
  return wait_all_bools(futs);
}

bool ib_communicator_t::expect_bytes(size_t num_bytes, std::vector<char> &out) {
  out.reserve(num_bytes);

  auto [success, _] = connection.recv_with_bytes(
    com_tag::coordinator_bcast_bytes,
    {(void*)out.data(), num_bytes}).get();

  return success;
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
