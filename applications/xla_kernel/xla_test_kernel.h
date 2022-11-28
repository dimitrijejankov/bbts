#pragma once

#include <cassert>
#include "../../main/ud_functions/xla_kernel_base.h"

namespace bbts {

struct xla_test_kernel : public xla_kernel_base {

  // initializes the function
  xla_test_kernel();

  const static std::string hlo;
};

}