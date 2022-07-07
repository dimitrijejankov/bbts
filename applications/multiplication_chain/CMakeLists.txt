# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(generate_multiplication_chain ${app_dir}/generate_multiplication_chain.cc)
set_target_properties(generate_multiplication_chain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_multiplication_chain ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_multiplication_chain ${MPI_LIBRARIES})
target_link_libraries(generate_multiplication_chain bbts-common)

# add the dependencies
add_dependencies(applications generate_multiplication_chain)
