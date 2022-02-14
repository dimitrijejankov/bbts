# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(ib ${app_dir}/ib.cc)
set_target_properties(ib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(ib ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(ib ${MPI_LIBRARIES})
target_link_libraries(ib bbts-common)

# add the dependencies
add_dependencies(applications ib)
