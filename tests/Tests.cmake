# enable tests
enable_testing()

# include the google test
include(GoogleTest)
find_package(GTest REQUIRED)

# get the current directory
get_filename_component(unit-test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile all the objects
if(${ENABLE_GPU})

# also add the .cu files
file(GLOB files "${unit-test-path}/*.cc" "${unit-test-path}/*.cu")

else()

# just use the .cc files
file(GLOB files "${unit-test-path}/*.cc")

endif()


# sorts files alphabetically because some tests require
# files created in previous tests
list(SORT files)
add_custom_target(unit-tests)
foreach(file ${files})

    # grab the name of the test without the extension
    get_filename_component(fileName "${file}" NAME_WE)

    # create the test executable
    add_executable(${fileName} ${file})

    # link the stuff we need
    target_link_libraries(${fileName} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${fileName} bbts-common)
    target_link_libraries(${fileName} ${MPI_LIBRARIES})
    target_compile_definitions(${fileName} PRIVATE -DGTEST_LINKED_AS_SHARED_LIBRARY )

    # enable cuda stuff
    set_target_properties(${fileName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # add the test as a dependency of the unit test target
    add_dependencies(unit-tests ${fileName})

endforeach()
