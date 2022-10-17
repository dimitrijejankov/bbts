#include "command.h"
#include "../utils/concurent_queue.h"
#include <condition_variable>
#include <mutex>
#include <queue>

namespace bbts {

// reduces (and partial reduces are given priority)
// if no reduces exist that we can run I pick an apply that is part of an reduce
// if not I pick an apply that will go into a reduce
// if not I pick the first apply available
class reorder_buffer_t {
public:

  reorder_buffer_t();

  void execute();

  void stop_executing();

  void shutdown();

  void clear();

  bool queue(command_ptr_t _command);

  bool get_next(command_t::op_type_t type, command_ptr_t &out);

private:

  // stuff to handle run and shutdown logic
  bool is_executing = false;
  bool _shutdown = false;
  std::mutex m;
  std::condition_variable cv;

  // stuff to handle applies and partial reduces
  std::queue<command_ptr_t> apply_queue;
  std::queue<command_ptr_t> partial_reduce_queue;
  std::mutex apply_reduce_m;
  std::condition_variable apply_reduce_cv;

  // stuff to handle distributed reduces
  std::mutex dist_reduce_m;
  std::condition_variable dist_reduce_cv;
  std::queue<command_ptr_t> reduce_queue;

  // stuff to handle moves and broadcasts
  std::mutex move_m;
  std::condition_variable move_cv;
  std::queue<command_ptr_t> move_queue;

  bool any_applies_or_partial_reduces();
};

}