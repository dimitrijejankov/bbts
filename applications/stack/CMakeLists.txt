# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(test_stack ${app_dir}/test_stack.cc)
set_target_properties(test_stack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(test_stack ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_stack ${MPI_LIBRARIES})
target_link_libraries(test_stack bbts-common)

# add the dependencies
add_dependencies(applications test_stack)
