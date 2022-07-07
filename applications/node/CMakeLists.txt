# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(node_cli ${app_dir}/node_cli.cc)
set_target_properties(node_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(node_cli ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(node_cli ${MPI_LIBRARIES})
target_link_libraries(node_cli bbts-common)

# add the dependencies
add_dependencies(applications node_cli)
