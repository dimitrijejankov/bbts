#include "../src/commands/reservation_station.h"
#include "../src/utils/concurent_queue.h"
#include <atomic>
#include <gtest/gtest.h>
#include <thread>

namespace bbts {

using _remote_cmd_t = std::tuple<command_ptr_t, std::atomic<bool> *>;

struct fake_storage_t {

  void insert(tid_t tid) {

    std::unique_lock<std::mutex> lck(_m);
    _sto.insert(tid);
  }

  void remove(tid_t tid) {
    std::unique_lock<std::mutex> lck(_m);
    _sto.erase(tid);
  }

  size_t num_tensors() const { return _sto.size(); }

  std::mutex _m;

  std::unordered_set<tid_t> _sto;
};

// the reservation station needs a deleter thread
std::thread create_deleter_thread(reservation_station_ptr_t &_rs,
                                  fake_storage_t &_sto, int32_t _num_to_del) {

  // create the thread

  return std::thread([_rs, &_sto, _num_to_del]() {
    // while we have something remove
    tid_t id;
    auto to_deleted = _num_to_del;
    while (true) {

      // check if done
      if (to_deleted == 0) {
        break;
      }

      // get the next tensor to remove
      id = _rs->get_to_remove();
      if (id == -1) {
        break;
      }

      // do some random sleeping
      usleep(rand() % 1000 + 100);

      // deleted
      _sto.remove(id);
      _rs->retire_remove(id);
      std::cout << "Remove tensor : " << id << '\n' << std::flush;
      to_deleted--;
    }
  });
}

std::thread create_remote_processing_thread(
    int32_t node, std::vector<reservation_station_ptr_t> &rss,
    std::vector<fake_storage_t> &sto,
    std::vector<bbts::concurent_queue<_remote_cmd_t>> &remote_cmds) {

  // create the thread to pull
  std::thread t = std::thread([&remote_cmds, node, &rss, &sto]() {
    // process the remote commands
    while (true) {

      // wait for the command
      _remote_cmd_t _rc;
      remote_cmds[node].wait_dequeue(_rc);

      // check if we are done...
      if (std::get<0>(_rc) == nullptr) {
        break;
      }

      // simulate the execution
      usleep(rand() % 100);

      // we only have remote moves here
      if (std::get<0>(_rc)->type == command_t::MOVE) {

        // get the output tensor
        auto [out_tid, out_node] = std::get<0>(_rc)->get_output(0);

        // make sure it is the same node
        EXPECT_EQ(node, out_node);
        std::cout << "MOVE from " << out_node << "\n" << std::flush;

        // mark that the tensor is
        sto[node].insert(out_tid);
      } else if (std::get<0>(_rc)->type == command_t::REDUCE) {

        // std::cout << "REDUCE from \n" << std::flush;
      }

      // retire the command
      rss[node]->retire_command(std::move(std::get<0>(_rc)));

      // mark that we are done
      *std::get<1>(_rc) = true;
    }
  });

  return std::move(t);
}

std::thread remote_tensor_notification_sender(
    int32_t my_node, int32_t out_node,
    std::vector<reservation_station_ptr_t> &rss,
    std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>>
        &remote_notifications) {
  // create the thread
  std::thread t =
      std::thread([my_node, out_node, &rss, &remote_notifications]() {
        while (true) {

          // get tensors to notify the other node
          bool is_done;
          auto tensors =
              rss[my_node]->tensors_to_notify_node(out_node, is_done);

          // if it is node break out
          if (is_done) {
            break;
          }

          // add the remote commands
          std::pair<std::vector<tid_t>, int32_t> to_store = {tensors, my_node};
          remote_notifications[out_node].enqueue(to_store);
        }
      });

  return std::move(t);
}

std::thread tensor_notifier(
    int32_t my_node, std::vector<reservation_station_ptr_t> &rss,
    std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>>
        &remote_notifications) {
  // create the thread
  std::thread t = std::thread([my_node, &rss, &remote_notifications]() {
    while (true) {

      // wait for the command
      std::pair<std::vector<tid_t>, int32_t> tensors;
      remote_notifications[my_node].wait_dequeue(tensors);

      // check if we are done...
      if (std::get<1>(tensors) == -1) {
        break;
      }

      // notify that the tensors became available
      rss[my_node]->notify_available_tensors(tensors.second, tensors.first);
    }
  });

  return std::move(t);
}

std::tuple<std::thread, std::thread, std::thread> create_command_processing_thread(
    std::vector<reservation_station_ptr_t> &rss,
    std::vector<fake_storage_t> &sto,
    std::vector<bbts::concurent_queue<_remote_cmd_t>> &remote_cmds,
    int32_t node) {

  // create the thread to pull
  std::thread t_move = std::thread([&rss, &remote_cmds, &sto, node]() {
    while (true) {

      // get the command
      auto cmd = rss[node]->get_next_move_command();
      if (cmd == nullptr) {
        break;
      }

      // move the
      std::cout << "MOVE " << cmd->id << " on my_node : " << node
                << " Executed...\n"
                << std::flush;

      // get the target node
      auto target = cmd->get_output(0);

      // we use this to busy wait
      std::atomic<bool> done{};
      done = false;

      // push the command
      _remote_cmd_t c = {cmd->clone(), &done};
      remote_cmds[target.node].enqueue(c);

      // add some waiting to simulate running the command
      usleep(rand() % 100);

      // retire the command
      rss[node]->retire_command(std::move(cmd));

      // wait for the remote command to be done
      while (!done) {
      }
    }
  });

  std::thread t_apply = std::thread([&rss, &remote_cmds, &sto, node]() {
    while (true) {

      // get the command
      auto cmd = rss[node]->get_next_apply_command();
      if (cmd == nullptr) {
        break;
      }

      std::cout << "APPLY " << cmd->id << " on my_node : " << node
                << " Executed...\n"
                << std::flush;

      if (cmd->get_root_node() == node) {

        // store the outputs so we can add it to the storage
        auto outputs = cmd->get_outputs();

        // update all the outputs in the storage
        for (auto &o : outputs) {
          sto[node].insert(o.tid);
        }

        // retire the command
        rss[node]->retire_command(std::move(cmd));
      } else {

        // all applies are local
        throw std::runtime_error("How did this happen!");
      }
    }
  });

  std::thread t_reduce = std::thread([&rss, &remote_cmds, &sto, node]() {
    while (true) {

      // get the command
      auto cmd = rss[node]->get_next_reduce_command();
      if (cmd == nullptr) {
        break;
      }

      // check if the reduce is remote or local
      if (cmd->is_local_reduce(node)) {

        // std::cout << "LOCAL_REDUCE " << cmd->id << " on node " << node <<
        // '\n' << std::flush;

        // store the outputs so we can add it to the storage
        auto outputs = cmd->get_outputs();

        // update all the outputs in the storage
        for (auto &o : outputs) {
          sto[node].insert(o.tid);
        }

        // retire the command
        rss[node]->retire_command(std::move(cmd));
      } else {

        std::cout << "REMOTE_REDUCE " << cmd->id << " on node " << node
                  << '\n'
                  << std::flush;

        // store the outputs so we can add it to the storage
        auto outputs = cmd->get_outputs();

        // update all the outputs in the storage
        for (auto &o : outputs) {
          sto[node].insert(o.tid);
        }

        // figure out the nodes we need to retire the command to
        std::unordered_set<node_id_t> nodes_to_retire;
        for (auto &in : cmd->get_inputs()) {
          nodes_to_retire.insert(in.node);
        }
        for (auto &out : cmd->get_outputs()) {
          nodes_to_retire.insert(out.node);
        }

        // we use this to busy wait
        std::vector<std::atomic<bool>> done(nodes_to_retire.size());
        for (auto &d : done) {
          d = false;
        }

        // retire the command for each node
        int32_t i = 0;
        for (auto n : nodes_to_retire) {

          if (n == node) {
            done[i++] = true;
            continue;
          }

          // push the command
          _remote_cmd_t c = {cmd->clone(), &done[i++]};
          remote_cmds[n].enqueue(c);
        }

        // add some waiting to simulate running the command
        usleep(rand() % 100);

        // retire the command
        rss[node]->retire_command(std::move(cmd));

        // wait till all done
        for (auto &d : done) {
          while (!d) {
          }
        }
      }
    }
  });

  return {std::move(t_move), std::move(t_apply), std::move(t_reduce)};
}

TEST(TestReservationStation, FewLocalCommands0) {

  auto cmd = command_t::create_apply(
      0, {0, 0}, false, {}, {command_t::tid_node_id_t{.tid = 0, .node = 0}},
      {command_t::tid_node_id_t{.tid = 2, .node = 0},
       command_t::tid_node_id_t{.tid = 3, .node = 0}});

  auto cmd2 = cmd->clone();

  auto inputs = cmd2->get_outputs();
  for (auto in : inputs) {
    std::cout << in.tid << "\n";
  }
}

TEST(TestReservationStation, FewLocalCommands1) {

  // tensors = { (0, 0), (1, 0) }
  // APPLY (.input = {(0, 0)}, .output = {(2, 0)})
  // DELETE (.input = {(0, 0)})
  // REDUCE (.input = {(1, 0), (2, 0)}, .output = {(3, 0)})
  // DELETE (.input = {(1, 0), (2, 0)})

  // create the storage
  fake_storage_t storage;

  // create two input tensors
  storage.insert(0);
  storage.insert(1);

  // create the reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // create the deleter thread
  auto deleter = create_deleter_thread(rs, storage, 3);

  // register the tensor
  rs->register_tensor(0);
  rs->register_tensor(1);

  // make a command that applies something to tensor 0
  EXPECT_TRUE(rs->queue_command(command_t::create_apply(
      0, {0, 0}, false, {}, {command_t::tid_node_id_t{.tid = 0, .node = 0}},
      {command_t::tid_node_id_t{.tid = 2, .node = 0}})));

  // make a command that deletes tensor with tid 0
  EXPECT_TRUE(rs->queue_command(command_t::create_delete(
      1, {command_t::tid_node_id_t{.tid = 0, .node = 0}})));

  // make a command that runs a reduce
  EXPECT_TRUE(rs->queue_command(command_t::create_reduce(
      2, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 1, .node = 0},
       command_t::tid_node_id_t{.tid = 2, .node = 0}},
      {command_t::tid_node_id_t{.tid = 3, .node = 0}})));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs->queue_command(command_t::create_delete(
      3, {command_t::tid_node_id_t{.tid = 1, .node = 0},
          command_t::tid_node_id_t{.tid = 2, .node = 0}})));

  // kick off commands
  rs->execute_scheduled_async();

  // get the first command to execute
  auto c1 = rs->get_next_apply_command();

  // retire the command as we pretend we have executed it
  storage.insert(2);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command
  auto c2 = rs->get_next_reduce_command();
  storage.insert(3);
  EXPECT_TRUE(rs->retire_command(std::move(c2)));

  // wait for stuff to finish
  deleter.join();

  // shutdown the reservation station
  rs->shutdown();

  // make sure there is only one tensors
  EXPECT_EQ(storage.num_tensors(), 1);

  // wait for the reservation station to finish
  rs->wait_until_finished();
}

TEST(TestReservationStation, FewLocalCommands2) {

  // tensors = { (0, 0), (1, 0) }
  // APPLY (.input = {(0, 0)}, .output = {(2, 0)})
  // REDUCE (.input = {(1, 0), (2, 0)}, .output = {(3, 0)})
  // DELETE (.input = {(0, 0), (1, 0), (2, 0), (3, 0)})

  // create the storage
  fake_storage_t storage;

  // create two input tensors
  storage.insert(0);
  storage.insert(1);

  // create the reservation station
  auto rs = std::make_shared<reservation_station_t>(0, 1);

  // create the deleter thread
  auto deleter = create_deleter_thread(rs, storage, 4);

  // register the tensor
  rs->register_tensor(0);
  rs->register_tensor(1);

  // make a command that applies something to tensor 0
  EXPECT_TRUE(rs->queue_command(command_t::create_apply(
      0, {0, 0}, false, {}, {command_t::tid_node_id_t{.tid = 0, .node = 0}},
      {command_t::tid_node_id_t{.tid = 2, .node = 0}})));

  // make a command that runs a reduce
  EXPECT_TRUE(rs->queue_command(command_t::create_reduce(
      1, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 1, .node = 0},
       command_t::tid_node_id_t{.tid = 2, .node = 0}},
      {command_t::tid_node_id_t{.tid = 3, .node = 0}})));

  // kick off commands
  rs->execute_scheduled_async();

  // get the first command to execute
  auto c1 = rs->get_next_apply_command();

  // retire the command as we pretend we have executed it
  storage.insert(2);
  EXPECT_TRUE(rs->retire_command(std::move(c1)));

  // get the next command
  auto c2 = rs->get_next_reduce_command();
  storage.insert(3);
  EXPECT_TRUE(rs->retire_command(std::move(c2)));

  // make a command that deletes all the tensors except for the tid = 3 tensor
  EXPECT_TRUE(rs->queue_command(command_t::create_delete(
      2, {command_t::tid_node_id_t{.tid = 0, .node = 0},
          command_t::tid_node_id_t{.tid = 1, .node = 0},
          command_t::tid_node_id_t{.tid = 2, .node = 0},
          command_t::tid_node_id_t{.tid = 3, .node = 0}})));

  // shutdown the reservation station
  rs->shutdown();

  // wait for stuff to finish
  deleter.join();

  // make sure there is only one tensors
  EXPECT_EQ(storage.num_tensors(), 0);

  // wait for the reservation station to finish
  rs->wait_until_finished();
}

TEST(TestReservationStation, TwoNodesBMM) {

  // create the storage
  std::vector<fake_storage_t> sto(2);

  // create the two reservation stations
  std::vector<reservation_station_ptr_t> rss;
  rss.push_back(std::make_shared<reservation_station_t>(0, 2));
  rss.push_back(std::make_shared<reservation_station_t>(1, 2));

  //       Tensors for A
  // | rowID | colID | tid | node |
  // |   0   |   0   |  0  |  1   |
  // |   0   |   1   |  1  |  0   |
  // |   1   |   0   |  2  |  0   |
  // |   1   |   1   |  3  |  1   |

  // (0, 0)
  rss[1]->register_tensor(0);
  sto[1].insert(0);

  // (0, 1)
  rss[0]->register_tensor(1);
  sto[0].insert(1);

  // (1, 0)
  rss[0]->register_tensor(2);
  sto[0].insert(2);

  // (1, 1)
  rss[1]->register_tensor(3);
  sto[1].insert(3);

  //       Tensors for B
  // | rowID | colID | tid | node |
  // |   0   |   0   |  4  |  1   |
  // |   0   |   1   |  5  |  1   |
  // |   1   |   0   |  6  |  0   |
  // |   1   |   1   |  7  |  0   |

  // (0, 0)
  rss[1]->register_tensor(4);
  sto[1].insert(4);

  // (1, 0)
  rss[1]->register_tensor(5);
  sto[1].insert(5);

  // (0, 1)
  rss[0]->register_tensor(6);
  sto[0].insert(6);

  // (1, 1)
  rss[0]->register_tensor(7);
  sto[0].insert(7);

  // we put the commands we want to schedule here
  std::vector<command_ptr_t> _cmds;

  /// 1.1 shuffle A.rowID

  // MOVE (.input = {(0, 1)}, .output = {(0, 0)})
  _cmds.emplace_back(
      command_t::create_move(0, command_t::tid_node_id_t{.tid = 0, .node = 1},
                             command_t::tid_node_id_t{.tid = 0, .node = 0}));

  // MOVE (.input = {(0, 2)}, .output = {(1, 2)})
  _cmds.emplace_back(
      command_t::create_move(1, command_t::tid_node_id_t{.tid = 2, .node = 0},
                             command_t::tid_node_id_t{.tid = 2, .node = 1}));

  /// 1.2 broadcast B

  // MOVE (.input = {(4, 1)}, .output = {(4, 0)})
  _cmds.emplace_back(
      command_t::create_move(2, command_t::tid_node_id_t{.tid = 4, .node = 1},
                             command_t::tid_node_id_t{.tid = 4, .node = 0}));

  // MOVE (.input = {(5, 1)}, .output = {(5, 0)})
  _cmds.emplace_back(
      command_t::create_move(3, command_t::tid_node_id_t{.tid = 5, .node = 1},
                             command_t::tid_node_id_t{.tid = 5, .node = 0}));

  // MOVE (.input = {(6, 0)}, .output = {(6, 1)})
  _cmds.emplace_back(
      command_t::create_move(4, command_t::tid_node_id_t{.tid = 6, .node = 0},
                             command_t::tid_node_id_t{.tid = 6, .node = 1}));

  // MOVE (.input = {(7, 0)}, .output = {(7, 1)})
  _cmds.emplace_back(
      command_t::create_move(5, command_t::tid_node_id_t{.tid = 7, .node = 0},
                             command_t::tid_node_id_t{.tid = 7, .node = 1}));

  /// 2.1 Do the multiply

  // (0, 0) x (0, 0) - APPLY (.input = {(0, 0), (4, 0)}, .output = {(8, 0)})
  _cmds.emplace_back(
      command_t::create_apply(6, {0, 0}, false, {},
                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                               command_t::tid_node_id_t{.tid = 4, .node = 0}},
                              {command_t::tid_node_id_t{.tid = 8, .node = 0}}));

  // (0, 1) x (1, 0) - APPLY (.input = {(1, 0), (6, 0)}, .output = {(9, 0)})
  _cmds.emplace_back(
      command_t::create_apply(7, {0, 0}, false, {},
                              {command_t::tid_node_id_t{.tid = 1, .node = 0},
                               command_t::tid_node_id_t{.tid = 6, .node = 0}},
                              {command_t::tid_node_id_t{.tid = 9, .node = 0}}));

  // (1, 0) x (0, 0) - APPLY (.input = {(2, 1), (4, 1)}, .output = {(10, 1)})
  _cmds.emplace_back(command_t::create_apply(
      8, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 2, .node = 1},
       command_t::tid_node_id_t{.tid = 4, .node = 1}},
      {command_t::tid_node_id_t{.tid = 10, .node = 1}}));

  // (1, 1) x (1, 0) - APPLY (.input = {(3, 1), (6, 1)}, .output = {(11, 1)})
  _cmds.emplace_back(command_t::create_apply(
      9, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 3, .node = 1},
       command_t::tid_node_id_t{.tid = 6, .node = 1}},
      {command_t::tid_node_id_t{.tid = 11, .node = 1}}));

  // (0, 0) x (0, 1) - APPLY (.input = {(0, 0), (5, 0)}, .output = {(12, 0)})
  _cmds.emplace_back(command_t::create_apply(
      10, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 0, .node = 0},
       command_t::tid_node_id_t{.tid = 5, .node = 0}},
      {command_t::tid_node_id_t{.tid = 12, .node = 0}}));

  // (0, 1) x (1, 1) - APPLY (.input = {(1, 0), (7, 0)}, .output = {(13, 0)})
  _cmds.emplace_back(command_t::create_apply(
      11, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 1, .node = 0},
       command_t::tid_node_id_t{.tid = 7, .node = 0}},
      {command_t::tid_node_id_t{.tid = 13, .node = 0}}));

  // (1, 0) x (0, 1) - APPLY (.input = {(2, 1), (5, 1)}, .output = {(14, 1)})
  _cmds.emplace_back(command_t::create_apply(
      12, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 2, .node = 1},
       command_t::tid_node_id_t{.tid = 5, .node = 1}},
      {command_t::tid_node_id_t{.tid = 14, .node = 1}}));

  // (1, 1) x (1, 1) - APPLY (.input = {(3, 1), (7, 1)}, .output = {(15, 1)})
  _cmds.emplace_back(command_t::create_apply(
      13, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 3, .node = 1},
       command_t::tid_node_id_t{.tid = 7, .node = 1}},
      {command_t::tid_node_id_t{.tid = 15, .node = 1}}));

  /// 2.2 Do the reduce

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(8, 0), (9, 0)},
  // .output = {(16, 0)})
  _cmds.emplace_back(command_t::create_reduce(
      14, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 8, .node = 0},
       command_t::tid_node_id_t{.tid = 9, .node = 0}},
      {command_t::tid_node_id_t{.tid = 16, .node = 0}}));

  // (1, 0) x (0, 0) + (1, 1) x (1, 0) - REDUCE (.input = {(10, 1), (11, 1)},
  // .output = {(17, 1)})
  _cmds.emplace_back(command_t::create_reduce(
      15, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 10, .node = 1},
       command_t::tid_node_id_t{.tid = 11, .node = 1}},
      {command_t::tid_node_id_t{.tid = 17, .node = 1}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(12, 0), (13, 0)},
  // .output = {(18, 0)})
  _cmds.emplace_back(command_t::create_reduce(
      16, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 12, .node = 0},
       command_t::tid_node_id_t{.tid = 13, .node = 0}},
      {command_t::tid_node_id_t{.tid = 18, .node = 0}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(14, 1), (15, 1)},
  // .output = {(19, 1)})
  _cmds.emplace_back(command_t::create_reduce(
      17, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 14, .node = 1},
       command_t::tid_node_id_t{.tid = 15, .node = 1}},
      {command_t::tid_node_id_t{.tid = 19, .node = 1}}));

  /// Remove the intermediate results

  // remove them from node 0
  _cmds.emplace_back(command_t::create_delete(
      18, {command_t::tid_node_id_t{.tid = 1, .node = 0},
           command_t::tid_node_id_t{.tid = 2, .node = 0},
           command_t::tid_node_id_t{.tid = 6, .node = 0},
           command_t::tid_node_id_t{.tid = 7, .node = 0},
           command_t::tid_node_id_t{.tid = 0, .node = 0},
           command_t::tid_node_id_t{.tid = 4, .node = 0},
           command_t::tid_node_id_t{.tid = 5, .node = 0},
           command_t::tid_node_id_t{.tid = 8, .node = 0},
           command_t::tid_node_id_t{.tid = 9, .node = 0},
           command_t::tid_node_id_t{.tid = 12, .node = 0},
           command_t::tid_node_id_t{.tid = 13, .node = 0}}));

  // remove them from node 1
  _cmds.emplace_back(command_t::create_delete(
      19, {command_t::tid_node_id_t{.tid = 0, .node = 1},
           command_t::tid_node_id_t{.tid = 3, .node = 1},
           command_t::tid_node_id_t{.tid = 4, .node = 1},
           command_t::tid_node_id_t{.tid = 5, .node = 1},
           command_t::tid_node_id_t{.tid = 2, .node = 1},
           command_t::tid_node_id_t{.tid = 6, .node = 1},
           command_t::tid_node_id_t{.tid = 7, .node = 1},
           command_t::tid_node_id_t{.tid = 10, .node = 1},
           command_t::tid_node_id_t{.tid = 11, .node = 1},
           command_t::tid_node_id_t{.tid = 14, .node = 1},
           command_t::tid_node_id_t{.tid = 15, .node = 1}}));

  // schedule them all at once
  for (auto &cmd : _cmds) {

    // if it uses node 0
    if (cmd->uses_node(0)) {
      EXPECT_TRUE(rss[0]->queue_command(cmd->clone()));
    }

    // check if node 1 uses the command
    if (cmd->uses_node(1)) {
      EXPECT_TRUE(rss[1]->queue_command(cmd->clone()));
    }
  }

  // kick off commands
  for (const auto &rs : rss) {
    rs->execute_scheduled_async();
  }

  // create the queues for commands
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(2);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  _remote_executor_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {

    // store the thread
    _remote_executor_threads.push_back(std::move(
        create_remote_processing_thread(node, rss, sto, remote_cmds)));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  _executor_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {

    auto [t_m, t_a, t_r] = create_command_processing_thread(rss, sto, remote_cmds, node);
    _executor_threads.push_back(std::move(t_m));
    _executor_threads.push_back(std::move(t_a));
    _executor_threads.push_back(std::move(t_r));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.push_back(std::move(create_deleter_thread(rss[0], sto[0], 11)));
  deleters.push_back(std::move(create_deleter_thread(rss[1], sto[1], 11)));

  // wait for the deleters
  deleters[0].join();
  deleters[1].join();

  // shutdown the rss
  rss[0]->shutdown();
  rss[1]->shutdown();

  _remote_cmd_t c = {nullptr, nullptr};
  remote_cmds[0].enqueue(c);
  remote_cmds[1].enqueue(c);

  // wait for remote executors to finish
  for (auto &t : _remote_executor_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _executor_threads) {
    t.join();
  }

  // make sure there are exactly two for the reduce...
  EXPECT_EQ(sto[0].num_tensors(), 2);
  EXPECT_EQ(sto[1].num_tensors(), 2);

  // wait for the reservation station to finish
  for (auto &rs : rss) {
    rs->wait_until_finished();
  }
}

TEST(TestReservationStation, TwoNodesCMM) {

  // create the storage
  std::vector<fake_storage_t> sto(2);

  std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>>
      remote_notifications(2);

  // create the two reservation stations
  std::vector<reservation_station_ptr_t> rss;
  rss.push_back(std::make_shared<reservation_station_t>(0, 2));
  rss.push_back(std::make_shared<reservation_station_t>(1, 2));

  //       Tensors for A
  // | rowID | colID | tid | node |
  // |   0   |   0   |  0  |  1   | MOVE TO NODE 0
  // |   0   |   1   |  1  |  0   | MOVE TO NODE 1
  // |   1   |   0   |  2  |  0   |
  // |   1   |   1   |  3  |  1   |

  // (0, 0)
  rss[1]->register_tensor(0);
  sto[1].insert(0);

  // (0, 1)
  rss[0]->register_tensor(1);
  sto[0].insert(1);

  // (1, 0)
  rss[0]->register_tensor(2);
  sto[0].insert(2);

  // (1, 1)
  rss[1]->register_tensor(3);
  sto[1].insert(3);

  //       Tensors for B
  // | rowID | colID | tid | node |
  // |   0   |   0   |  4  |  1   | MOVE TO 0
  // |   0   |   1   |  5  |  0   |
  // |   1   |   0   |  6  |  1   |
  // |   1   |   1   |  7  |  0   | MOVE TO 1

  // (0, 0)
  rss[1]->register_tensor(4);
  sto[1].insert(4);

  // (1, 0)
  rss[0]->register_tensor(5);
  sto[0].insert(5);

  // (0, 1)
  rss[1]->register_tensor(6);
  sto[1].insert(6);

  // (1, 1)
  rss[0]->register_tensor(7);
  sto[0].insert(7);

  // we put the commands we want to schedule here
  std::vector<command_ptr_t> _cmds;

  /// 1.1 shuffle A.colID

  // MOVE (.input = {( 0,  1)}, .output = {( 0,  0)})
  _cmds.emplace_back(
      command_t::create_move(0, command_t::tid_node_id_t{.tid = 0, .node = 1},
                             command_t::tid_node_id_t{.tid = 0, .node = 0}));

  // MOVE (.input = {(1,  0)}, .output = {(1,  1)})
  _cmds.emplace_back(
      command_t::create_move(1, command_t::tid_node_id_t{.tid = 1, .node = 0},
                             command_t::tid_node_id_t{.tid = 1, .node = 1}));

  /// 1.2 shuffle B.rowID

  // MOVE (.input = {(4, 1)}, .output = {(4, 0)})
  _cmds.emplace_back(
      command_t::create_move(2, command_t::tid_node_id_t{.tid = 4, .node = 1},
                             command_t::tid_node_id_t{.tid = 4, .node = 0}));

  // MOVE (.input = {(7, 0)}, .output = {(7, 1)})
  _cmds.emplace_back(
      command_t::create_move(3, command_t::tid_node_id_t{.tid = 7, .node = 0},
                             command_t::tid_node_id_t{.tid = 7, .node = 1}));

  /// 2.1 Do the multiply

  // (0, 0) x (0, 0) - APPLY (.input = {(0, 0), (4, 0)}, .output = {(8, 0)})
  _cmds.emplace_back(
      command_t::create_apply(4, {0, 0}, false, {},
                              {command_t::tid_node_id_t{.tid = 0, .node = 0},
                               command_t::tid_node_id_t{.tid = 4, .node = 0}},
                              {command_t::tid_node_id_t{.tid = 8, .node = 0}}));

  // (0, 1) x (1, 0) - APPLY (.input = {(1, 1), (6, 1)}, .output = {(9, 1)})
  _cmds.emplace_back(
      command_t::create_apply(5, {0, 0}, false, {},
                              {command_t::tid_node_id_t{.tid = 1, .node = 1},
                               command_t::tid_node_id_t{.tid = 6, .node = 1}},
                              {command_t::tid_node_id_t{.tid = 9, .node = 1}}));

  // (1, 0) x (0, 0) - APPLY (.input = {(2, 0), (4, 0)}, .output = {(10, 0)})
  _cmds.emplace_back(command_t::create_apply(
      6, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 2, .node = 0},
       command_t::tid_node_id_t{.tid = 4, .node = 0}},
      {command_t::tid_node_id_t{.tid = 10, .node = 0}}));

  // (1, 1) x (1, 0) - APPLY (.input = {(3, 1), (6, 1)}, .output = {(11, 1)})
  _cmds.emplace_back(command_t::create_apply(
      7, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 3, .node = 1},
       command_t::tid_node_id_t{.tid = 6, .node = 1}},
      {command_t::tid_node_id_t{.tid = 11, .node = 1}}));

  // (0, 0) x (0, 1) - APPLY (.input = {(0, 0), (5, 0)}, .output = {(12, 0)})
  _cmds.emplace_back(command_t::create_apply(
      8, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 0, .node = 0},
       command_t::tid_node_id_t{.tid = 5, .node = 0}},
      {command_t::tid_node_id_t{.tid = 12, .node = 0}}));

  // (0, 1) x (1, 1) - APPLY (.input = {(1, 0), (7, 0)}, .output = {(13, 0)})
  _cmds.emplace_back(command_t::create_apply(
      9, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 1, .node = 1},
       command_t::tid_node_id_t{.tid = 7, .node = 1}},
      {command_t::tid_node_id_t{.tid = 13, .node = 1}}));

  // (1, 0) x (0, 1) - APPLY (.input = {(2, 1), (5, 1)}, .output = {(14, 1)})
  _cmds.emplace_back(command_t::create_apply(
      10, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 2, .node = 0},
       command_t::tid_node_id_t{.tid = 5, .node = 0}},
      {command_t::tid_node_id_t{.tid = 14, .node = 0}}));

  // (1, 1) x (1, 1) - APPLY (.input = {(3, 1), (7, 1)}, .output = {(15, 1)})
  _cmds.emplace_back(command_t::create_apply(
      11, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 3, .node = 1},
       command_t::tid_node_id_t{.tid = 7, .node = 1}},
      {command_t::tid_node_id_t{.tid = 15, .node = 1}}));

  /// 2.2 Do the reduce

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(8, 0), (9, 1)},
  // .output = {(16, 0)})
  _cmds.emplace_back(command_t::create_reduce(
      12, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 8, .node = 0},
       command_t::tid_node_id_t{.tid = 9, .node = 1}},
      {command_t::tid_node_id_t{.tid = 16, .node = 0}}));

  // (1, 0) x (0, 0) + (1, 1) x (1, 0) - REDUCE (.input = {(10, 0), (11, 1)},
  // .output = {(17, 1)})
  _cmds.emplace_back(command_t::create_reduce(
      13, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 10, .node = 0},
       command_t::tid_node_id_t{.tid = 11, .node = 1}},
      {command_t::tid_node_id_t{.tid = 17, .node = 1}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(12, 0), (13, 0)},
  // .output = {(18, 0)})
  _cmds.emplace_back(command_t::create_reduce(
      14, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 12, .node = 0},
       command_t::tid_node_id_t{.tid = 13, .node = 1}},
      {command_t::tid_node_id_t{.tid = 18, .node = 0}}));

  // (0, 0) x (0, 0) + (0, 1) x (1, 0) - REDUCE (.input = {(14, 1), (15, 1)},
  // .output = {(19, 1)})
  _cmds.emplace_back(command_t::create_reduce(
      15, {0, 0}, false, {},
      {command_t::tid_node_id_t{.tid = 14, .node = 0},
       command_t::tid_node_id_t{.tid = 15, .node = 1}},
      {command_t::tid_node_id_t{.tid = 19, .node = 1}}));

  /// 3.0 Remove the intermediate results

  // remove them from node 0
  _cmds.emplace_back(command_t::create_delete(
      16, {command_t::tid_node_id_t{.tid = 1, .node = 0},
           command_t::tid_node_id_t{.tid = 2, .node = 0},
           command_t::tid_node_id_t{.tid = 5, .node = 0},
           command_t::tid_node_id_t{.tid = 7, .node = 0},
           command_t::tid_node_id_t{.tid = 0, .node = 0},
           command_t::tid_node_id_t{.tid = 4, .node = 0},
           command_t::tid_node_id_t{.tid = 8, .node = 0},
           command_t::tid_node_id_t{.tid = 10, .node = 0},
           command_t::tid_node_id_t{.tid = 12, .node = 0},
           command_t::tid_node_id_t{.tid = 14, .node = 0},
           command_t::tid_node_id_t{.tid = 16, .node = 0},
           command_t::tid_node_id_t{.tid = 18, .node = 0}}));

  // remove them from node 0
  _cmds.emplace_back(command_t::create_delete(
      17, {command_t::tid_node_id_t{.tid = 0, .node = 1},
           command_t::tid_node_id_t{.tid = 3, .node = 1},
           command_t::tid_node_id_t{.tid = 4, .node = 1},
           command_t::tid_node_id_t{.tid = 6, .node = 1},
           command_t::tid_node_id_t{.tid = 1, .node = 1},
           command_t::tid_node_id_t{.tid = 7, .node = 1},
           command_t::tid_node_id_t{.tid = 9, .node = 1},
           command_t::tid_node_id_t{.tid = 11, .node = 1},
           command_t::tid_node_id_t{.tid = 13, .node = 1},
           command_t::tid_node_id_t{.tid = 15, .node = 1},
           command_t::tid_node_id_t{.tid = 17, .node = 1},
           command_t::tid_node_id_t{.tid = 19, .node = 1}}));

  // schedule them all at once
  for (auto &cmd : _cmds) {

    // if it uses node 0
    if (cmd->uses_node(0)) {
      EXPECT_TRUE(rss[0]->queue_command(cmd->clone()));
    }

    // check if node 1 uses the command
    if (cmd->uses_node(1)) {
      EXPECT_TRUE(rss[1]->queue_command(cmd->clone()));
    }
  }

  // kick off commands
  for (const auto &rs : rss) {
    rs->execute_scheduled_async();
  }

  // create the queues for commands
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(2);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  _remote_executor_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {

    // store the thread
    _remote_executor_threads.push_back(std::move(
        create_remote_processing_thread(node, rss, sto, remote_cmds)));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  _executor_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {

    auto [t_m, t_a, t_r] = create_command_processing_thread(rss, sto, remote_cmds, node);
    _executor_threads.push_back(std::move(t_m));
    _executor_threads.push_back(std::move(t_a));
    _executor_threads.push_back(std::move(t_r));
  }

  //
  std::vector<std::thread> _notifier_threads;
  _notifier_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {
    for (node_id_t out_node = 0; out_node < 2; ++out_node) {
      _notifier_threads.push_back(std::move(remote_tensor_notification_sender(
          node, out_node, rss, remote_notifications)));
    }
  }

  //
  std::vector<std::thread> _notification_handler_threads;
  _notification_handler_threads.reserve(2);
  for (node_id_t node = 0; node < 2; ++node) {
    _notification_handler_threads.push_back(
        std::move(tensor_notifier(node, rss, remote_notifications)));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.push_back(std::move(create_deleter_thread(rss[0], sto[0], 12)));
  deleters.push_back(std::move(create_deleter_thread(rss[1], sto[1], 12)));

  // wait for the deleters
  deleters[0].join();
  deleters[1].join();

  // shutdown the rss
  rss[0]->shutdown();
  rss[1]->shutdown();

  _remote_cmd_t c = {nullptr, nullptr};
  remote_cmds[0].enqueue(c);
  remote_cmds[1].enqueue(c);

  std::pair<std::vector<tid_t>, int32_t> d = {{}, -1};
  remote_notifications[0].enqueue(d);
  remote_notifications[1].enqueue(d);

  // wait for remote executors to finish
  for (auto &t : _remote_executor_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _executor_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _notifier_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _notification_handler_threads) {
    t.join();
  }

  // make sure there are exactly two for the reduce...
  EXPECT_EQ(sto[0].num_tensors(), 0);
  EXPECT_EQ(sto[1].num_tensors(), 0);

  // wait for the reservation station to finish
  for (auto &rs : rss) {
    rs->wait_until_finished();
  }
}

std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>>
init_matrix(size_t split, size_t num_nodes, tid_t &cur_tid,
            std::vector<fake_storage_t> &sto,
            std::vector<reservation_station_ptr_t> &rss,
            std::vector<std::vector<int32_t>> &to_del) {

  std::map<std::tuple<int32_t, int32_t>, std::tuple<node_id_t, tid_t>> mat_a;
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the node
      node_id_t node = rand() % num_nodes;
      mat_a[{rowID, colID}] = {node, cur_tid};

      if (cur_tid == 250) {
        std::cout << "bla\n";
      }
      // register the tensor
      rss[node]->register_tensor(cur_tid);
      sto[node].insert(cur_tid);

      // mark that we need to delete it later
      to_del[node].push_back(cur_tid);

      // go to the next tid
      cur_tid++;
    }
  }

  return std::move(mat_a);
}

template <class fun>
void create_shuffle(size_t num_nodes, size_t split, command_id_t &cur_cmd,
                    fun fn,
                    std::map<std::tuple<int32_t, int32_t>,
                             std::tuple<node_id_t, tid_t>> &mat_locs,
                    std::vector<command_ptr_t> &_cmds,
                    std::vector<std::vector<int32_t>> &to_del) {

  // do the shuffle
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // get the tid and the node of this block
      auto &[node, tid] = mat_locs[{rowID, colID}];

      // no need to move here
      auto target_node = (node_id_t)fn(rowID, colID, num_nodes);
      if (node == target_node) {
        continue;
      }

      if (tid == 250) {
        std::cout << "bla\n";
      }

      // move it
      _cmds.emplace_back(command_t::create_move(
          cur_cmd++, command_t::tid_node_id_t{.tid = tid, .node = node},
          command_t::tid_node_id_t{.tid = tid, .node = target_node}));

      // mark that we need to delete it later
      to_del[target_node].push_back(tid);
    }
  }
}

TEST(TestReservationStation, NNodesCMM) {

  size_t num_nodes = 5;
  size_t split = 25;

  command_id_t cur_cmd = 0;
  tid_t cur_tid = 0;

  // all the tensors that we need to delete
  std::vector<std::vector<int32_t>> to_del(num_nodes);

  // create the storage
  std::vector<bbts::concurent_queue<std::pair<std::vector<tid_t>, int32_t>>>
      remote_notifications(num_nodes);
  std::vector<fake_storage_t> sto(num_nodes);

  // create the two reservation stations
  std::vector<reservation_station_ptr_t> rss(num_nodes);
  for (auto node = 0; node < num_nodes; ++node) {
    rss[node] = std::make_shared<reservation_station_t>(node, num_nodes);
  }

  // init the two matrices
  auto a_mat = init_matrix(split, num_nodes, cur_tid, sto, rss, to_del);
  auto b_mat = init_matrix(split, num_nodes, cur_tid, sto, rss, to_del);

  // we put the commands we want to schedule here
  std::vector<command_ptr_t> _cmds;

  // do the shuffle for a
  create_shuffle(
      num_nodes, split, cur_cmd,
      [](int32_t rowID, int32_t colID, size_t num_nodes) {
        return colID % num_nodes;
      },
      a_mat, _cmds, to_del);

  // do the shuffle for b
  create_shuffle(
      num_nodes, split, cur_cmd,
      [](int32_t rowID, int32_t colID, size_t num_nodes) {
        return rowID % num_nodes;
      },
      b_mat, _cmds, to_del);

  // create all the multiply commands
  std::map<std::tuple<int32_t, int32_t>,
           std::vector<std::tuple<tid_t, node_id_t>>>
      multiplies;
  for (int32_t i = 0; i < split; ++i) {
    for (int32_t j = 0; j < split; ++j) {
      for (int32_t k = 0; k < split; ++k) {

        // get the tid and the node
        auto [a_node, a_tid] = a_mat[{i, k}];
        auto [b_node, b_tid] = b_mat[{k, j}];

        // get the target node
        auto target_node = (node_id_t)(k % num_nodes);

        // add the command
        _cmds.emplace_back(command_t::create_apply(
            cur_cmd++, {0, 0}, false, {},
            {command_t::tid_node_id_t{.tid = a_tid, .node = target_node},
             command_t::tid_node_id_t{.tid = b_tid, .node = target_node}},
            {command_t::tid_node_id_t{.tid = cur_tid, .node = target_node}}));

        to_del[target_node].push_back(cur_tid);
        multiplies[{i, j}].push_back({cur_tid, target_node});
        cur_tid++;
      }
    }
  }

  // create the aggregate
  for (int32_t rowID = 0; rowID < split; ++rowID) {
    for (int32_t colID = 0; colID < split; ++colID) {

      // all the multiplied tensors
      auto &muls = multiplies[{rowID, colID}];

      // get the target node
      auto target_node = (node_id_t)((rowID + colID * split) % num_nodes);

      // figure out the inputs
      std::vector<bbts::command_t::tid_node_id_t> inputs;
      for (auto &mul : muls) {
        auto &[tid, node] = mul;
        inputs.push_back({.tid = tid, .node = node});
      }

      // create the reduce command
      _cmds.emplace_back(command_t::create_reduce(
          cur_cmd++, {0, 0}, false, {}, inputs,
          {command_t::tid_node_id_t{.tid = cur_tid, .node = target_node}}));

      cur_tid++;
    }
  }

  // prepare the removes
  for (int32_t node = 0; node < num_nodes; ++node) {

    // store the number we need to delete
    std::vector<bbts::command_t::tid_node_id_t> _inputs;
    _inputs.reserve(to_del[node].size());
    for (auto t : to_del[node]) {
      _inputs.push_back(command_t::tid_node_id_t{.tid = t, .node = node});
    }

    // remove them from node
    _cmds.emplace_back(command_t::create_delete(cur_cmd++, _inputs));
  }

  // schedule them all at once
  for (auto &_cmd : _cmds) {

    // if it uses node 0
    for (int32_t node = 0; node < num_nodes; ++node) {
      if (_cmd->uses_node(node)) {
        EXPECT_TRUE(rss[node]->queue_command(_cmd->clone()));
      }
    }
  }

  // kick off commands
  for (const auto &rs : rss) {
    rs->execute_scheduled_async();
  }

  // create the queues for commands
  std::vector<bbts::concurent_queue<_remote_cmd_t>> remote_cmds(num_nodes);

  // these threads will process the remote move operations
  std::vector<std::thread> _remote_executor_threads;
  _remote_executor_threads.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {

    // store the thread
    _remote_executor_threads.push_back(std::move(
        create_remote_processing_thread(node, rss, sto, remote_cmds)));
  }

  // simulator threads
  std::vector<std::thread> _executor_threads;
  _executor_threads.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {

    auto [t_m, t_a, t_r] = create_command_processing_thread(rss, sto, remote_cmds, node);
    _executor_threads.push_back(std::move(t_m));
    _executor_threads.push_back(std::move(t_a));
    _executor_threads.push_back(std::move(t_r));
  }

  //
  std::vector<std::thread> _notifier_threads;
  _notifier_threads.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {
    for (node_id_t out_node = 0; out_node < num_nodes; ++out_node) {
      _notifier_threads.push_back(std::move(remote_tensor_notification_sender(
          node, out_node, rss, remote_notifications)));
    }
  }

  //
  std::vector<std::thread> _notification_handler_threads;
  _notification_handler_threads.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {
    _notification_handler_threads.push_back(
        std::move(tensor_notifier(node, rss, remote_notifications)));
  }

  // create the deleters
  std::vector<std::thread> deleters;
  deleters.reserve(num_nodes);
  for (node_id_t node = 0; node < num_nodes; ++node) {
    deleters.push_back(std::move(
        create_deleter_thread(rss[node], sto[node], to_del[node].size())));
  }

  // wait for the deleters
  for (node_id_t node = 0; node < num_nodes; ++node) {
    deleters[node].join();
  }

  // shutdown the rss
  for (node_id_t node = 0; node < num_nodes; ++node) {
    rss[node]->shutdown();
  }

  _remote_cmd_t c = {nullptr, nullptr};
  for (node_id_t node = 0; node < num_nodes; ++node) {
    remote_cmds[node].enqueue(c);
  }

  std::pair<std::vector<tid_t>, int32_t> d = {{}, -1};
  for (node_id_t node = 0; node < num_nodes; ++node) {
    remote_notifications[node].enqueue(d);
  }

  // wait for remote executors to finish
  for (auto &t : _remote_executor_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _executor_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _notifier_threads) {
    t.join();
  }

  // wait for the executors
  for (auto &t : _notification_handler_threads) {
    t.join();
  }

  size_t num = 0;
  for (node_id_t node = 0; node < num_nodes; ++node) {
    num += sto[node].num_tensors();
  }

  EXPECT_EQ(num, split * split);

  // wait for the reservation station to finish
  for (auto &rs : rss) {
    rs->wait_until_finished();
  }
}

} // namespace bbts
