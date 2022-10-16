#include "command.h"
#include "../utils/concurent_queue.h"
#include <mutex>

namespace bbts {

// reduces (and partial reduces are given priority)
// if no reduces exist that we can run I pick an apply that is part of an reduce
// if not I pick an apply that will go into a reduce
// if not I pick the first apply available
class heuristic_t {
public:

  heuristic_t();

  void execute();

  void stop_executing();

  void shutdown();

  void clear();

  bool queue(command_ptr_t _command);

  bool get_next(command_t::op_type_t type, command_ptr_t &out);

  concurent_queue<command_ptr_t> apply_queue;
  concurent_queue<command_ptr_t> reduce_queue;
  concurent_queue<command_ptr_t> move_queue;

  bool is_executing = false;

  std::mutex m;

  std::condition_variable cv;
};

}