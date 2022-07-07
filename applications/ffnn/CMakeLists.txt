get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_ffnn ${app_dir}/generate_ffnn.cc)
set_target_properties(generate_ffnn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_ffnn ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_ffnn ${MPI_LIBRARIES})
target_link_libraries(generate_ffnn bbts-common)



add_library(ffnn_lib SHARED ${app_dir}/ffnn_lib_init.cc  
                            ${app_dir}/ffnn_activation_mult.cc  
                            ${app_dir}/ffnn_mult.cc      
                            ${app_dir}/ffnn_uniform_data.cc
                            ${app_dir}/ffnn_add.cc              
                            ${app_dir}/ffnn_uniform_weights.cc
                            ${app_dir}/ffnn_types.cc            
                            ${app_dir}/ffnn_weighted_sum.cc
                            ${app_dir}/ffnn_matrix_hadamard.cc  
                            ${app_dir}/ffnn_back_mult.cc  
                            ${app_dir}/ffnn_uniform_sparse_data.cc  
                            ${app_dir}/ffnn_weighted_sum_sparse_dense.cc)


# add the dependencies
add_dependencies(applications generate_ffnn)
