#pragma once

#include <iostream>

namespace bbts {

// represents a raw vector
template<class T>
struct raw_vector_t {

  // return the data
  const T &operator[](size_t idx) const { return _data[idx]; };

  // the size
  size_t size() const { return _num_elements; }

  // is the vector empty
  bool empty() const { return _num_elements == 0; }

  // return the data
  const T *data() const { return _data; }

  // the iterator for the raw vector
  struct raw_vector_iterator_t {

    // all the relevant functions
    raw_vector_iterator_t(const T *value = nullptr) : _p(value) {}
    raw_vector_iterator_t &operator++() {
      _p++;
      return *this;
    }
    raw_vector_iterator_t operator++(int) {
      int *temp = _p;
      _p++;
      return raw_vector_iterator_t(temp);
    }
    bool operator==(raw_vector_iterator_t other) const { return _p == other._p; }
    bool operator!=(raw_vector_iterator_t other) const { return _p != other._p; }
    const T &operator*() const { return *_p; }

    size_t distance_from(const raw_vector_iterator_t &start) const {
      return (((size_t) _p - (size_t) start._p)) / sizeof(T);
    }

   private:

    // pointer to the current element
    const T *_p;
  };

  // finds the value
  raw_vector_iterator_t find(const T &value) const {

    // find it
    for (auto it = begin(); it != end(); ++it) {
      if (*it == value) {
        return it;
      }
    }
    return end();
  }

  raw_vector_iterator_t begin() const { return raw_vector_iterator_t(_data); }
  raw_vector_iterator_t end() const { return raw_vector_iterator_t(_data + _num_elements); }

  // data
  const T *_data;

  // then number of tensors in the list
  size_t _num_elements;
};

}