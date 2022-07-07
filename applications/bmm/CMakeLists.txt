get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_bmm ${app_dir}/generate_bmm.cc)
set_target_properties(generate_bmm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_bmm ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_bmm ${MPI_LIBRARIES})
target_link_libraries(generate_bmm bbts-common)




# add the dependencies
add_dependencies(applications generate_bmm)
