# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dimitrije/bbts

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dimitrije/bbts

# Include any dependencies generated for this target.
include CMakeFiles/test_parsed_command.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_parsed_command.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_parsed_command.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_parsed_command.dir/flags.make

CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o: CMakeFiles/test_parsed_command.dir/flags.make
CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o: tests/test_parsed_command.cc
CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o: CMakeFiles/test_parsed_command.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o -MF CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o.d -o CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o -c /home/dimitrije/bbts/tests/test_parsed_command.cc

CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/tests/test_parsed_command.cc > CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.i

CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/tests/test_parsed_command.cc -o CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.s

# Object files for target test_parsed_command
test_parsed_command_OBJECTS = \
"CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o"

# External object files for target test_parsed_command
test_parsed_command_EXTERNAL_OBJECTS =

CMakeFiles/test_parsed_command.dir/cmake_device_link.o: CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: CMakeFiles/test_parsed_command.dir/build.make
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/lib/libgtest.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/lib/libgtest_main.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: libbbts-common.a
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpicxx.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpifort.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /opt/intel/oneapi/mpi/2021.6.0/lib/release/libmpi.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/lib64/libdl.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/lib64/librt.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/lib64/libpthread.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_rt.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/local/lib/libprotobuf.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: /usr/local/cuda/lib64/libcublas.so
CMakeFiles/test_parsed_command.dir/cmake_device_link.o: CMakeFiles/test_parsed_command.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test_parsed_command.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_parsed_command.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_parsed_command.dir/build: CMakeFiles/test_parsed_command.dir/cmake_device_link.o
.PHONY : CMakeFiles/test_parsed_command.dir/build

# Object files for target test_parsed_command
test_parsed_command_OBJECTS = \
"CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o"

# External object files for target test_parsed_command
test_parsed_command_EXTERNAL_OBJECTS =

bin/test_parsed_command: CMakeFiles/test_parsed_command.dir/tests/test_parsed_command.cc.o
bin/test_parsed_command: CMakeFiles/test_parsed_command.dir/build.make
bin/test_parsed_command: /usr/lib/libgtest.so
bin/test_parsed_command: /usr/lib/libgtest_main.so
bin/test_parsed_command: libbbts-common.a
bin/test_parsed_command: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpicxx.so
bin/test_parsed_command: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpifort.so
bin/test_parsed_command: /opt/intel/oneapi/mpi/2021.6.0/lib/release/libmpi.so
bin/test_parsed_command: /usr/lib64/libdl.so
bin/test_parsed_command: /usr/lib64/librt.so
bin/test_parsed_command: /usr/lib64/libpthread.so
bin/test_parsed_command: /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_rt.so
bin/test_parsed_command: /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so
bin/test_parsed_command: /usr/local/lib/libprotobuf.so
bin/test_parsed_command: /usr/local/cuda/lib64/libcublas.so
bin/test_parsed_command: CMakeFiles/test_parsed_command.dir/cmake_device_link.o
bin/test_parsed_command: CMakeFiles/test_parsed_command.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bin/test_parsed_command"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_parsed_command.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_parsed_command.dir/build: bin/test_parsed_command
.PHONY : CMakeFiles/test_parsed_command.dir/build

CMakeFiles/test_parsed_command.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_parsed_command.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_parsed_command.dir/clean

CMakeFiles/test_parsed_command.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/test_parsed_command.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_parsed_command.dir/depend
