get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(gpu_bench ${app_dir}/bench_matrix_mutiply.cc)
set_target_properties(gpu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(gpu_bench ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gpu_bench ${MPI_LIBRARIES})
target_link_libraries(gpu_bench bbts-common)

# add the dependencies
add_dependencies(applications gpu_bench)
