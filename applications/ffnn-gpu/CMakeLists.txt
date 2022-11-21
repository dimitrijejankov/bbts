get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_gpu_ffnn ${app_dir}/generate_ffnn.cc)
set_target_properties(generate_gpu_ffnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_gpu_ffnn ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_gpu_ffnn ${MPI_LIBRARIES})
target_link_libraries(generate_gpu_ffnn bbts-common)

add_library(ffnn_gpu_lib SHARED ${app_dir}/ffnn_lib_init.cc  
                                ${app_dir}/ffnn_activation_mult.cu  
                                ${app_dir}/ffnn_add.cu              
                                ${app_dir}/ffnn_mult.cc      
                                ${app_dir}/ffnn_uniform_data.cc
                                ${app_dir}/ffnn_uniform_weights.cc
                                ${app_dir}/ffnn_types.cc    
                                ${app_dir}/ffnn_diff_mm_kernel.cu        
                                ${app_dir}/ffnn_weighted_sum.cu
                                ${app_dir}/ffnn_matrix_hadamard.cu  
                                ${app_dir}/ffnn_back_mult.cu  
                                ${app_dir}/ffnn_uniform_sparse_data.cc  
                                ${app_dir}/ffnn_weighted_sum_sparse_dense.cc)
target_link_libraries(ffnn_gpu_lib CUDA::curand)  

# add the dependencies
add_dependencies(applications generate_gpu_ffnn)
