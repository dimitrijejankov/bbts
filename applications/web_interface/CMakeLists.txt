# create the application executable
get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

add_executable(web_interface ${app_dir}/run_web.cc)
set_target_properties(web_interface PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(web_interface ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(web_interface ${MPI_LIBRARIES})
target_link_libraries(web_interface bbts-common)

# add the dependencies
add_dependencies(applications web_interface)
