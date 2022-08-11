#include "../../src/gpu/scheduler.h"
#include "../../src/tensor/builtin_formats.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using matrix_index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<bbts::tid_t, float>>;
using matrix_reduce_index_t = std::map<std::tuple<int32_t, int32_t>, std::tuple<float, std::vector<bbts::tid_t>>>;

void init_tensor_on_cpu(const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                        const bbts::tensor_factory_ptr_t &factory,
                        const bbts::storage_ptr_t &storage, bbts::tid_t tid,
                        uint32_t num_rows, uint32_t num_cols, float value);

void init_blocked_matrix(float &val,
                         bbts::tid_t &cur_idx,
                         matrix_index_t &index, 
                         const bbts::multi_gpu_scheduler_ptr_t &scheduler, 
                         const bbts::tensor_factory_ptr_t &factory,
                         const bbts::storage_ptr_t &storage,
                         size_t row_blocking, 
                         size_t row_block_size,
                         size_t col_blocking, 
                         size_t col_block_size);

bbts::command_ptr_t
create_apply(bbts::command_id_t id,
             bbts::udf_manager_ptr udm, const std::string &ud_name,
             const std::vector<bbts::tid_t> &inputs,
             const std::vector<bbts::tid_t> &outputs,
             const std::vector<bbts::command_param_t> &params);

bbts::command_ptr_t
create_reduce(bbts::command_id_t id,
              bbts::udf_manager_ptr udm, const std::string &ud_name,
              const std::vector<bbts::tid_t> &inputs,
              bbts::tid_t output,
              const std::vector<bbts::command_param_t> &params);

bbts::command_ptr_t
create_delete(bbts::command_id_t id, const std::vector<bbts::tid_t> &inputs);

std::vector<bbts::command_ptr_t> make_multiply(bbts::tid_t &cur_tid,
                                               bbts::udf_manager_ptr udf_manager,
                                               matrix_index_t &a_index, 
                                               matrix_index_t &b_index, 
                                               matrix_index_t &c_index, 
                                               size_t matrix_blocking, 
                                               size_t matrix_block_size);


std::vector<bbts::command_ptr_t> delete_matrix(matrix_index_t &index);