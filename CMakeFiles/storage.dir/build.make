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
include CMakeFiles/storage.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/storage.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/storage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/storage.dir/flags.make

CMakeFiles/storage.dir/src/storage/memory_storage.cc.o: CMakeFiles/storage.dir/flags.make
CMakeFiles/storage.dir/src/storage/memory_storage.cc.o: src/storage/memory_storage.cc
CMakeFiles/storage.dir/src/storage/memory_storage.cc.o: CMakeFiles/storage.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/storage.dir/src/storage/memory_storage.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/storage.dir/src/storage/memory_storage.cc.o -MF CMakeFiles/storage.dir/src/storage/memory_storage.cc.o.d -o CMakeFiles/storage.dir/src/storage/memory_storage.cc.o -c /home/dimitrije/bbts/src/storage/memory_storage.cc

CMakeFiles/storage.dir/src/storage/memory_storage.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/storage.dir/src/storage/memory_storage.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/src/storage/memory_storage.cc > CMakeFiles/storage.dir/src/storage/memory_storage.cc.i

CMakeFiles/storage.dir/src/storage/memory_storage.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/storage.dir/src/storage/memory_storage.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/src/storage/memory_storage.cc -o CMakeFiles/storage.dir/src/storage/memory_storage.cc.s

CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o: CMakeFiles/storage.dir/flags.make
CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o: src/storage/nvme_storage.cc
CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o: CMakeFiles/storage.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o -MF CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o.d -o CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o -c /home/dimitrije/bbts/src/storage/nvme_storage.cc

CMakeFiles/storage.dir/src/storage/nvme_storage.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/storage.dir/src/storage/nvme_storage.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/src/storage/nvme_storage.cc > CMakeFiles/storage.dir/src/storage/nvme_storage.cc.i

CMakeFiles/storage.dir/src/storage/nvme_storage.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/storage.dir/src/storage/nvme_storage.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/src/storage/nvme_storage.cc -o CMakeFiles/storage.dir/src/storage/nvme_storage.cc.s

storage: CMakeFiles/storage.dir/src/storage/memory_storage.cc.o
storage: CMakeFiles/storage.dir/src/storage/nvme_storage.cc.o
storage: CMakeFiles/storage.dir/build.make
.PHONY : storage

# Rule to build all files generated by this target.
CMakeFiles/storage.dir/build: storage
.PHONY : CMakeFiles/storage.dir/build

CMakeFiles/storage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/storage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/storage.dir/clean

CMakeFiles/storage.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/storage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/storage.dir/depend

