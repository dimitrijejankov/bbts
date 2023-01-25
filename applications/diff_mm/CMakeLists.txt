get_filename_component(app_dir ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# create the application executable
add_executable(generate_diff_mm ${app_dir}/generate_diff_mm.cc)
set_target_properties(generate_diff_mm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# link the stuff we need
target_link_libraries(generate_diff_mm ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(generate_diff_mm ${MPI_LIBRARIES})
target_link_libraries(generate_diff_mm bbts-common)


add_library(diff_mm_lib SHARED ${app_dir}/diff_mm_lib_init.cc  
                               ${app_dir}/diff_mm_kernel.cc  
                               ${app_dir}/generate_diff_mm.cc)

# add the dependencies
add_dependencies(applications generate_diff_mm)
