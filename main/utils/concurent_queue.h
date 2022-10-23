//
// Created by dimitrije on 3/29/19.
//

#pragma once

#include <cassert>
#include <cstdint>
#include <queue>
#include <condition_variable>

namespace bbts {

#define CONCURENT_QUEUE_YES 1
#define CONCURENT_QUEUE_NO 0
#define CONCURENT_QUEUE_SHUTDOWN -1

template <typename T>
class concurent_queue {

 private:

  // the queue implementation
  std::queue<T> _internal_queue;

  // the mutex to lock the structure
  std::mutex _m;

  // the conditional variable to wait
  std::condition_variable _cv;

  // is this queue shutdown
  bool _shutdown = false;

 public:

  inline void shutdown() {
    
    // notify all that we have shutdown
    std::unique_lock<std::mutex> lk(_m);
    _shutdown = true;
    _cv.notify_all();
  }

  inline void clear() {
    std::unique_lock<std::mutex> lk(_m);
    _internal_queue = {};
  }

  inline bool wait_dequeue(T &item) {

    // wait until we have something in the queue
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{return _shutdown || !_internal_queue.empty();});

    // if we have shutdown now finish
    if(_shutdown && _internal_queue.empty()) { return false; }

    // grab the element and pop the queue
    item = std::move(_internal_queue.front());
    _internal_queue.pop();
    return true;
  };

  inline int try_dequeue(T &item) {

    std::unique_lock<std::mutex> lk(_m);

    // if we have shutdown now finish
    if(_shutdown && _internal_queue.empty()) { return CONCURENT_QUEUE_SHUTDOWN; }

    // we have nothing go wait a bit
    if(_internal_queue.empty()) { return CONCURENT_QUEUE_NO; }

    // grab the element and pop the queue
    item = std::move(_internal_queue.front());
    _internal_queue.pop();
    return CONCURENT_QUEUE_YES;
  }

  inline bool wait_dequeue_all(std::vector<T> &container) {

    // wait until we have something in the queue
    assert(container.empty());
    std::unique_lock<std::mutex> lk(_m);
    _cv.wait(lk, [&]{return _shutdown || !_internal_queue.empty();});

    // if we have shutdown now finish
    if(_shutdown && _internal_queue.empty()) { return false; }

    // grab the element and pop the queue
    size_t idx = 0;
    container.resize(_internal_queue.size());
    while (!_internal_queue.empty()) {
      container[idx++] = std::move(_internal_queue.front());
      _internal_queue.pop();
    }
    return true;
  };

  inline void enqueue(T& item) {

    // wait for lock
    std::unique_lock<std::mutex> lk(_m);

    // insert the element in the queue
    _internal_queue.push(std::move(item));

    // notify all the threads that are waiting
    _cv.notify_all();
  }

  inline void enqueue_copy(T item) {

    // wait for lock
    std::unique_lock<std::mutex> lk(_m);

    // insert the element in the queue
    _internal_queue.push(std::move(item));

    // notify all the threads that are waiting
    _cv.notify_all();
  }
};

}
