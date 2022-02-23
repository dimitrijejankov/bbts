#pragma once

//#include "mpi_communicator.h"
#include "ib_communicator.h"

namespace bbts {

//using communicator_t = mpi_communicator_t;
//using communicator_ptr_t = std::shared_ptr<mpi_communicator_t>;

using communicator_t = ib_communicator_t;
using communicator_ptr_t = std::shared_ptr<ib_communicator_t>;

}
