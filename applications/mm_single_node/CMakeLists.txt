get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_single_node_mm ${app_dir}/generate_single_node_mm.cc)
set_target_properties(generate_single_node_mm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_single_node_mm ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_single_node_mm ${MPI_LIBRARIES})
target_link_libraries(generate_single_node_mm bbts-common)


# add the dependencies
add_dependencies(applications generate_single_node_mm)
