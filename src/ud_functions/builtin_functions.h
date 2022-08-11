#pragma once

#include "ud_function.h"
#include "impls/dense_matrix_add.h"
#include "impls/dense_matrix_gpu_add.h"
#include "impls/dense_matrix_mult.h"
#include "impls/dense_matrix_gpu_mult.h"
#include "impls/dense_uniform.h"
#include "impls/dense_const_gpu.h"

namespace bbts {

/// 1. Matrix addition
ud_func_ptr_t get_matrix_add_udf();

/// 2. Matrix multiply
ud_func_ptr_t get_matrix_mult_udf();

/// 3. Uniform distribution
ud_func_ptr_t get_matrix_uniform_udf();

/// 4. Const initialization
ud_func_ptr_t get_matrix_const_udf();

}
