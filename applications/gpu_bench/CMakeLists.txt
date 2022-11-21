get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application for matrix multiplication benchmark
add_executable(gpu_mm_bench ${app_dir}/bench_matrix_mutiply.cc ${app_dir}/mm_util.cc)
set_target_properties(gpu_mm_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(gpu_mm_bench ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gpu_mm_bench ${MPI_LIBRARIES})
target_link_libraries(gpu_mm_bench bbts-common)

# add the dependencies
add_dependencies(applications gpu_mm_bench)

# create the application for matrix multiplication benchmark
add_executable(gpu_mm_chain_bench ${app_dir}/bench_mm_chain.cc ${app_dir}/mm_util.cc)
set_target_properties(gpu_mm_chain_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(gpu_mm_chain_bench ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gpu_mm_chain_bench ${MPI_LIBRARIES})
target_link_libraries(gpu_mm_chain_bench bbts-common)

# add the dependencies
add_dependencies(applications gpu_mm_chain_bench)

# create the application for ffnn benchmark
add_executable(gpu_ffnn_bench ${app_dir}/bench_ffnn.cc)
set_target_properties(gpu_ffnn_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(gpu_ffnn_bench ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(gpu_ffnn_bench ${MPI_LIBRARIES})
target_link_libraries(gpu_ffnn_bench bbts-common)

# add the dependencies
add_dependencies(applications bench_ffnn_diff)

# create the application for ffnn benchmark
add_executable(bench_ffnn_diff ${app_dir}/bench_ffnn_diff.cc)
set_target_properties(bench_ffnn_diff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(bench_ffnn_diff ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(bench_ffnn_diff ${MPI_LIBRARIES})
target_link_libraries(bench_ffnn_diff bbts-common)

# add the dependencies
add_dependencies(applications bench_ffnn_diff)
