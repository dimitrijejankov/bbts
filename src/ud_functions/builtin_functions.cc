#include "builtin_functions.h"

/// 1. Matrix Add
bbts::ud_func_ptr_t bbts::get_matrix_add_udf() {
  return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "matrix_add",
          .is_ass = true,
          .is_com = true,
          .num_in = 2,
          .num_out = 1,
          .impls = {},
      }
  );
}

/// 2. Matrix Multiply
bbts::ud_func_ptr_t bbts::get_matrix_mult_udf() {
  return std::make_unique<ud_func_t>(
      ud_func_t {
          .ud_name = "matrix_mult",
          .is_ass = true,
          .is_com = false,
          .num_in = 2,
          .num_out = 1,
          .impls = {},
      }
  );
}

/// 3. Uniform distribution
bbts::ud_func_ptr_t bbts::get_matrix_uniform_udf() {
  return std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
          .ud_name = "uniform",
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {},
      }
  );
}

/// 4. Const distribution
bbts::ud_func_ptr_t bbts::get_matrix_const_udf() {
  return std::make_unique<bbts::ud_func_t>(
      bbts::ud_func_t {
          .ud_name = "const",
          .is_ass = false,
          .is_com = false,
          .num_in = 0,
          .num_out = 1,
          .impls = {},
      }
  );
}
