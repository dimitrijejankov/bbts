get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_cpmm ${app_dir}/generate_cpmm.cc)
set_target_properties(generate_cpmm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_cpmm ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_cpmm ${MPI_LIBRARIES})
target_link_libraries(generate_cpmm bbts-common)

# add the dependencies
add_dependencies(applications generate_cpmm)
