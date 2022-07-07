#pragma once

#include "gpu_scheduler.h"
#include "../tensor/tensor_factory.h"
#include "../ud_functions/builtin_functions.h"
#include "../server/static_config.h"

namespace bbts {

// given a bunch of implementations find the udf that works
class udf_matcher {

public:

  // the matches
  udf_matcher(std::vector<ud_impl_ptr_t> &impls);

  // try to find a match for the via the input strings
  // TODO this is most likely going to be slow but for now it is fine
  ud_impl_t* findMatch(const std::vector<std::string> &inputs,
                       const std::vector<std::string> &outputs,
                       bool require_gpu = false);

 private:

  // the implementations we want to match to
  std::vector<ud_impl_ptr_t> &_impls;
};
using udf_matcher_ptr = std::unique_ptr<udf_matcher>;

//
class udf_manager_t {
public:

  // initializes the UD manager
  udf_manager_t(tensor_factory_ptr_t _tensor_factory, 
                gpu_scheduler_ptr_t _gpu_scheduler);

  // registers a udf with the system
  ud_id_t register_udf(ud_func_ptr_t _fun);

  // registers the implementation for a udf with the system
  ud_impl_id_t register_udf_impl(ud_impl_ptr_t _impl);

  // returns a matcher for the given ud function that will be used to figure out
  udf_matcher_ptr get_matcher_for(const std::string &ud_name);

  udf_matcher_ptr get_matcher_for(const std::string &ud_name, bool is_commutative, bool is_associative);

  // return the implementation
  ud_impl_t* get_fn_impl(ud_impl_id_t _id);

private:

  // the GPU scheduler
  gpu_scheduler_ptr_t _gpu_scheduler;

  // we use this to grab types from strings
  tensor_factory_ptr_t _tensor_factory;

  // all the registered udfs
  std::vector<ud_func_ptr_t> _udfs;

  // maps the udf name to the impl_id
  std::unordered_map<std::string, ud_id_t> _udfs_name_to_id;
};

using udf_manager_ptr = std::shared_ptr<udf_manager_t>;

}
