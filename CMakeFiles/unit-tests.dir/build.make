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

# Utility rule file for unit-tests.

# Include any custom commands dependencies for this target.
include CMakeFiles/unit-tests.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/unit-tests.dir/progress.make

unit-tests: CMakeFiles/unit-tests.dir/build.make
.PHONY : unit-tests

# Rule to build all files generated by this target.
CMakeFiles/unit-tests.dir/build: unit-tests
.PHONY : CMakeFiles/unit-tests.dir/build

CMakeFiles/unit-tests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/unit-tests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/unit-tests.dir/clean

CMakeFiles/unit-tests.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/unit-tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/unit-tests.dir/depend
