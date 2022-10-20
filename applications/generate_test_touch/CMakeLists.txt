get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(test_touch ${app_dir}/generate_test_touch.cc)
set_target_properties(test_touch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(test_touch ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_touch ${MPI_LIBRARIES})
target_link_libraries(test_touch bbts-common)




# add the dependencies
add_dependencies(applications test_touch)
