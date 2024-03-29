#include "command.h"
#include "../utils/concurent_queue.h"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace bbts {

class apply_reorder_queue_t {
public:

  void queue(command_id_t apply_id);

  void analyze(const std::vector<command_ptr_t> &cmds);

  void reduce_started(command_id_t reduce_id);

  void clear();

  bool has_any();

  command_id_t get_next();

  std::set<command_id_t> applies_not_into_reduces;

  std::set<command_id_t> applies_into_reduces;

  std::set<command_id_t> applies_into_ongoing_reduces;

  // these are used in the reorder_buffer_cmp_t to compare
  std::unordered_set<command_id_t> _reduces_in_progress;

  // these are initialized once during the analize and cleared when clear is called
  std::unordered_map<command_id_t, command_id_t> _applies_into_reduce;
  std::unordered_multimap<command_id_t, command_id_t> _reduce_to_applies;
};

// reduces (and partial reduces are given priority)
// if no reduces exist that we can run I pick an apply that is part of an reduce
// if not I pick an apply that will go into a reduce
// if not I pick the first apply available
class reorder_buffer_t {
public:

  reorder_buffer_t();

  void analyze(const std::vector<command_ptr_t> &cmds);

  void execute();

  void stop_executing();

  void shutdown();

  void clear();

  void queue(command_ptr_t _command);

  bool get_next(command_t::op_type_t type, command_ptr_t &out);

private:

  void _queue_apply(command_ptr_t _command);
  void _queue_reduce(command_ptr_t _command);
  void _queue_partial_reduce(command_ptr_t _command);
  void _queue_move(command_ptr_t _command);

  // stuff to handle run and shutdown logic
  bool is_executing = false;
  bool _shutdown = false;
  std::mutex m;
  std::condition_variable cv;

  // we use these to lock the below structures
  std::mutex apply_reduce_m;
  std::condition_variable apply_reduce_cv;

  // stuff to handle applies and partial reduces
  std::queue<command_ptr_t> partial_reduce_queue;

  // the 
  apply_reorder_queue_t apply_reorder_queue;
  std::unordered_map<command_id_t, command_ptr_t> _applies;

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

using reorder_buffer_ptr = std::shared_ptr<reorder_buffer_t>;

}