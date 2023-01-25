#include "dense_matrix_gpu_add.h"
#include "../../tensor/builtin_formats.h"

bbts::dense_matrix_gpu_add_t::dense_matrix_gpu_add_t() {

    // set the names
    impl_name = "dense_matrix_add";
    ud_name = "matrix_add";

    // set the input and output types
    inputTypes = {"dense", "dense"};
    outputTypes = {"dense"};

    // both inputs zero and one can be used as the inplace output
    inputInplace = {};

    // this is a CPU dense add
    is_gpu = true;

    // set the function that actually performs the add
    fn = &dense_matrix_gpu_add_t::add;
}

size_t bbts::dense_matrix_gpu_add_t::get_complexity_hint(const bbts::ud_impl_t::tensor_params_t &params,
                                                    const bbts::ud_impl_t::meta_args_t &_in) {

    // O(n * m)
    const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();
    return m_a.num_rows * m_a.num_cols;
}

size_t bbts::dense_matrix_gpu_add_t::get_required_memory(const bbts::ud_impl_t::tensor_params_t &params,
                                                         const bbts::ud_impl_t::meta_args_t &_in) const {
  return 0;
}

void bbts::dense_matrix_gpu_add_t::get_out_meta(const bbts::ud_impl_t::tensor_params_t &params,
                                                const bbts::ud_impl_t::meta_args_t &_in,
                                                bbts::ud_impl_t::meta_args_t &_out) const {

    // get the input argeters
    const auto &m_a = _in.get<0>().as<dense_tensor_meta_t>().m();

    // get the output argeters
    auto &m_out = _out.get<0>().as<dense_tensor_meta_t>().m();

    // set the new values
    m_out = { m_a.num_rows, m_a.num_cols };
}

// kernel definition
__global__ void dense_add_kernel(float *a, float *b, float *c, int n) {

    // get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void bbts::dense_matrix_gpu_add_t::add(const bbts::ud_impl_t::tensor_params_t &params,
                                       const bbts::ud_impl_t::tensor_args_t &_in,
                                       bbts::ud_impl_t::tensor_args_t &_out) {

    // get the tensors as dense tensors
    dense_tensor_t &a = _in.get<0>().as<dense_tensor_t>();
    dense_tensor_t &b = _in.get<1>().as<dense_tensor_t>();
    dense_tensor_t &out = _out.get<0>().as<dense_tensor_t>();

    // get the meta for the out tensor
    dense_tensor_meta_t &m_out = out.meta();

    // get the number of rows and columns
    uint32_t I = a.meta().m().num_rows;
    uint32_t J = a.meta().m().num_cols;


    // number of threads in each thread block
    uint32_t block_size = 1024;
 
    // number of thread blocks in grid
    uint32_t n = I * J;
    uint32_t grid_size = (int) ceil ((float) n / block_size);
 
    // Execute the kernel
    dense_add_kernel<<<grid_size, block_size, 0, params.stream>>>(a.data(), b.data(), out.data(), n);

    // set the new meta data
    m_out.m() = {I, J};
}