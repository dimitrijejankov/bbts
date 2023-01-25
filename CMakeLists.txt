cmake_minimum_required(VERSION 3.15)
project(tra_sub_sys_barbatos LANGUAGES CXX)

# set the c++ and cuda version
set(CMAKE_CXX_STANDARD 20)

# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "${PROJECT_SOURCE_DIR}/bin")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/libraries")

# set the location for cmake scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# link the std::filesystem
link_libraries(stdc++fs)

# if we are compiling for debug add the debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-D_GLIBCXX_DEBUG")
    endif()
    set(CMAKE_CXX_FLAGS "-g3 -fPIC -mavx")
    set(CMAKE_C_FLAGS "-g3 -fPIC -mavx")

    # some stuff in the linker to make .so loading nicer
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -ldl" )
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -ldl")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -ldl")

    # set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    # set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    # set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
    # set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")

# if we are compiling for release
elseif(CMAKE_BUILD_TYPE MATCHES Release)

    # do -O3 and set the instruction set to the native one for this machine
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC")

    # some stuff in the linker to make .so loading nicer
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -ldl" )
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -ldl")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -rdynamic -ldl")

endif()

# forward the definitions ENABLE_GPU=ON/OFF
if(${ENABLE_GPU})

    # enable the cuda language
    enable_language(CUDA)

    # set the cuda standard
    set(CMAKE_CUDA_STANDARD 17)

    # set the cuda flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread -Xcompiler -fPIC")
    
    # try to find cuda
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit)

    # set the right architecture
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

    # add the include directory
    include_directories("${CUDA_INCLUDE_DIRS}")

    # add the definition
    add_compile_definitions(ENABLE_GPU)
    
    # auto cuda paths
    file(GLOB UD_SOURCE_CUDA "main/ud_functions/impls/*.cu")
    set (UD_SOURCE_CUDA ${UD_SOURCE_CUDA} "main/ud_functions/gpu_scheduler_impl.cc")
endif()

# find the protocol buffer
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# forward the definitions ENABLE_HOOKS=ON/OFF
if(${ENABLE_HOOKS})
    add_compile_definitions(ENABLE_HOOKS)
endif()

# forward the definitions ENABLE_STORAGE=ON/OFF
if(${ENABLE_STORAGE})
    add_compile_definitions(ENABLE_STORAGE)
endif()

# try to find mkl
set(MKL_LINK sdl)
# set(MKL_LINK static)
# set(MKL_THREADING sequential)
find_package(MKL CONFIG REQUIRED)

# try to find the mpi library
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
endif(MPI_FOUND)

# the blas implementation
include_directories(${MKL_INCLUDE})

# include the third party stuff
include_directories(third-party)

# the commands
file(GLOB COMMANDS_SOURCE "main/commands/*.cc" "main/commands/command_handlers/*.cc")
add_library(commands OBJECT ${COMMANDS_SOURCE})

# the operations
file(GLOB OPERATIONS_SOURCE "main/operations/*.cc")
add_library(operations OBJECT ${OPERATIONS_SOURCE})

# the tensors
file(GLOB TENSOR_SOURCE "main/tensor/*.cc")
add_library(tensor OBJECT ${TENSOR_SOURCE})

# the storage
file(GLOB STORAGE_SOURCE "main/storage/*.cc")
add_library(storage OBJECT ${STORAGE_SOURCE})

# the server
file(GLOB SERVER_SOURCE "main/server/*.cc")
add_library(server OBJECT ${SERVER_SOURCE})

# the server
if(${ENABLE_GPU})
file(GLOB GPU_SOURCE "main/gpu/*.cc")
add_library(gpu OBJECT ${GPU_SOURCE})
endif()

# the communication
file(GLOB COMMUNICATION_SOURCE "main/communication/*.cc")
add_library(communication OBJECT ${COMMUNICATION_SOURCE})

# the ud functions
file(GLOB UD_SOURCE "main/ud_functions/*.cc" "main/ud_functions/impls/*.cc")
list(REMOVE_ITEM UD_SOURCE "main/ud_functions/gpu_scheduler_impl.cc")
add_library(ud_functions OBJECT ${UD_SOURCE} ${UD_SOURCE_CUDA})

# the protobuf 
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS main/gpu/gpu_profiler.proto)
add_library(proto_buf OBJECT ${PROTO_SRCS} ${PROTO_HDRS})

# link all modules up
add_library(bbts-common $<TARGET_OBJECTS:tensor>
                        $<TARGET_OBJECTS:commands>
                        $<TARGET_OBJECTS:operations>
                        $<TARGET_OBJECTS:storage>
                        $<TARGET_OBJECTS:communication>
                        $<TARGET_OBJECTS:server>
                        $<TARGET_OBJECTS:ud_functions>
                        $<TARGET_OBJECTS:proto_buf>
                        $<TARGET_OBJECTS:gpu>)

# link the GPU library
# if(${ENABLE_GPU})
# target_link_libraries(bbts-common $<TARGET_OBJECTS:gpu>)
# endif()

# to open .so files and stuff
target_link_libraries(bbts-common ${CMAKE_DL_LIBS})
target_link_libraries(bbts-common MKL::MKL)
target_link_libraries(bbts-common ${PROTOBUF_LIBRARIES})

# link the cublas library
if(${ENABLE_GPU})
    cuda_add_cublas_to_target(bbts-common)
endif()

# add the tests
include(tests/Tests.cmake)
include(integration_tests/Integration.cmake)
include(applications/Applications.cmake)

# enable einkorn if necesasry
if(${ENABLE_EINKORN})

# build einkorn
add_custom_target(build_einkorn 
                  COMMAND cabal build VERBATIM
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/einkorn/)
                  
add_custom_target(copy_einkorn 
                  COMMAND cp ${PROJECT_SOURCE_DIR}/third_party/einkorn/dist/build/toBarbaTos/toBarbaTos ${PROJECT_SOURCE_DIR}/bin VERBATIM)

# add dependencies
add_dependencies(build_einkorn copy_einkorn)
add_dependencies(node_cli build_einkorn)

# add the definition for einkorn
add_compile_definitions(ENABLE_EINKORN)

endif()