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
include CMakeFiles/test_move.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_move.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_move.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_move.dir/flags.make

CMakeFiles/test_move.dir/integration_tests/test_move.cc.o: CMakeFiles/test_move.dir/flags.make
CMakeFiles/test_move.dir/integration_tests/test_move.cc.o: integration_tests/test_move.cc
CMakeFiles/test_move.dir/integration_tests/test_move.cc.o: CMakeFiles/test_move.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_move.dir/integration_tests/test_move.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_move.dir/integration_tests/test_move.cc.o -MF CMakeFiles/test_move.dir/integration_tests/test_move.cc.o.d -o CMakeFiles/test_move.dir/integration_tests/test_move.cc.o -c /home/dimitrije/bbts/integration_tests/test_move.cc

CMakeFiles/test_move.dir/integration_tests/test_move.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_move.dir/integration_tests/test_move.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/integration_tests/test_move.cc > CMakeFiles/test_move.dir/integration_tests/test_move.cc.i

CMakeFiles/test_move.dir/integration_tests/test_move.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_move.dir/integration_tests/test_move.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/integration_tests/test_move.cc -o CMakeFiles/test_move.dir/integration_tests/test_move.cc.s

# Object files for target test_move
test_move_OBJECTS = \
"CMakeFiles/test_move.dir/integration_tests/test_move.cc.o"

# External object files for target test_move
test_move_EXTERNAL_OBJECTS =

bin/test_move: CMakeFiles/test_move.dir/integration_tests/test_move.cc.o
bin/test_move: CMakeFiles/test_move.dir/build.make
bin/test_move: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpicxx.so
bin/test_move: /opt/intel/oneapi/mpi/2021.6.0/lib/libmpifort.so
bin/test_move: /opt/intel/oneapi/mpi/2021.6.0/lib/release/libmpi.so
bin/test_move: /usr/lib64/libdl.so
bin/test_move: /usr/lib64/librt.so
bin/test_move: /usr/lib64/libpthread.so
bin/test_move: libbbts-common.a
bin/test_move: /opt/intel/oneapi/mkl/2022.1.0/lib/intel64/libmkl_rt.so
bin/test_move: /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so
bin/test_move: /usr/local/lib/libprotobuf.so
bin/test_move: /usr/local/cuda/lib64/libcublas.so
bin/test_move: CMakeFiles/test_move.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/test_move"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_move.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_move.dir/build: bin/test_move
.PHONY : CMakeFiles/test_move.dir/build

CMakeFiles/test_move.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_move.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_move.dir/clean

CMakeFiles/test_move.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/test_move.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_move.dir/depend
