#pragma once

#include <vector>
#include <cstdint>
#include <sstream>      
#include <functional>

namespace bbts {

  // the identifier of the tensor
  using tid_t = int32_t;

  // a tid reserved for when no tid is associated with any tensor
  const tid_t TID_NONE = -1;

  //  the identifier of the tensor format
  using tfid_t = int32_t;

  // holds the meta data of the tensor
  struct tensor_meta_t {

    // the meta is 256 by default
    static const size_t MAX_META_SIZE = 256;

    // we use this to cast between meta
    template<class T>
    T& as() const { return *((T*) this); }

    // the impl_id of the tensor format, it must me one of the values registered with the system
    tfid_t fmt_id;

    // the indices of the tensor
    char _blob[MAX_META_SIZE];
  };

  // the tensor class in our system
  struct tensor_t {
    
    // the constructors
    tensor_t() = default;
    tensor_t(void *data) { data_ptr = data; }

    // we can not copy a tensor as it will have the blob after
    tensor_t(const tensor_t&) = delete;
    void operator=(const tensor_t&) = delete;

    // we use this to cast the tensor into the actual implementation
    template<class T>
    T& as() const { return *((T*) this); }

    // returns the actual data
    template<class T> 
    T& get_data() const { 
      return *get_data_ptr<T>();  
    }

    // returns the point to the data
    template<class T> 
    T* get_data_ptr() const { 
      return (T*) (data_ptr == nullptr ? _blob : data_ptr);  
    }

    // returns the meta data
    template<class T> 
    T& get_meta() const { return _meta.as<T>(); }

  private:

    // the meta of the tensor
    tensor_meta_t _meta;
    
    // if the data pointer is set 
    void *data_ptr = nullptr;

    // after this tensor header is all the
    char _blob[0];
  };

  // to register a tensor format these need to be defined
  struct tensor_creation_fs_t {

    // returns the size of the tensor
    std::function<size_t(const tensor_meta_t&)> get_size;

    // initializes the tensor given some meta data to the provided memory, it should not use
    // any additional memory besides the one provided
    std::function<tensor_t&(void* here, const tensor_meta_t&)> init_tensor;

    // prints the content of a tensor
    std::function<void(const void* here, std::stringstream &ss)> print;
  };
}