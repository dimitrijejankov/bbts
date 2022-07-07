# get the current directory
get_filename_component(integration-test-path ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)

# compile all the objects
file(GLOB files "${integration-test-path}/*.cc")

# sorts files alphabetically because some tests require
# files created in previous tests
list(SORT files)
add_custom_target(integration-tests)
foreach(file ${files})

    # grab the name of the test without the extension
    get_filename_component(fileName "${file}" NAME_WE)

    # create the test executable
    add_executable(${fileName} ${file})

    # link the stuff we need
    target_link_libraries(${fileName} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${fileName} ${MPI_LIBRARIES})
    target_link_libraries(${fileName} bbts-common)

    # add the dependencies
    add_dependencies(integration-tests ${fileName})

endforeach()