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
include CMakeFiles/commands.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/commands.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/commands.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/commands.dir/flags.make

CMakeFiles/commands.dir/src/commands/command_runner.cc.o: CMakeFiles/commands.dir/flags.make
CMakeFiles/commands.dir/src/commands/command_runner.cc.o: src/commands/command_runner.cc
CMakeFiles/commands.dir/src/commands/command_runner.cc.o: CMakeFiles/commands.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/commands.dir/src/commands/command_runner.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/commands.dir/src/commands/command_runner.cc.o -MF CMakeFiles/commands.dir/src/commands/command_runner.cc.o.d -o CMakeFiles/commands.dir/src/commands/command_runner.cc.o -c /home/dimitrije/bbts/src/commands/command_runner.cc

CMakeFiles/commands.dir/src/commands/command_runner.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/commands.dir/src/commands/command_runner.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/src/commands/command_runner.cc > CMakeFiles/commands.dir/src/commands/command_runner.cc.i

CMakeFiles/commands.dir/src/commands/command_runner.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/commands.dir/src/commands/command_runner.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/src/commands/command_runner.cc -o CMakeFiles/commands.dir/src/commands/command_runner.cc.s

CMakeFiles/commands.dir/src/commands/reservation_station.cc.o: CMakeFiles/commands.dir/flags.make
CMakeFiles/commands.dir/src/commands/reservation_station.cc.o: src/commands/reservation_station.cc
CMakeFiles/commands.dir/src/commands/reservation_station.cc.o: CMakeFiles/commands.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/commands.dir/src/commands/reservation_station.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/commands.dir/src/commands/reservation_station.cc.o -MF CMakeFiles/commands.dir/src/commands/reservation_station.cc.o.d -o CMakeFiles/commands.dir/src/commands/reservation_station.cc.o -c /home/dimitrije/bbts/src/commands/reservation_station.cc

CMakeFiles/commands.dir/src/commands/reservation_station.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/commands.dir/src/commands/reservation_station.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/src/commands/reservation_station.cc > CMakeFiles/commands.dir/src/commands/reservation_station.cc.i

CMakeFiles/commands.dir/src/commands/reservation_station.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/commands.dir/src/commands/reservation_station.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/src/commands/reservation_station.cc -o CMakeFiles/commands.dir/src/commands/reservation_station.cc.s

CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o: CMakeFiles/commands.dir/flags.make
CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o: src/commands/tensor_notifier.cc
CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o: CMakeFiles/commands.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o -MF CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o.d -o CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o -c /home/dimitrije/bbts/src/commands/tensor_notifier.cc

CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/src/commands/tensor_notifier.cc > CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.i

CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/src/commands/tensor_notifier.cc -o CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.s

commands: CMakeFiles/commands.dir/src/commands/command_runner.cc.o
commands: CMakeFiles/commands.dir/src/commands/reservation_station.cc.o
commands: CMakeFiles/commands.dir/src/commands/tensor_notifier.cc.o
commands: CMakeFiles/commands.dir/build.make
.PHONY : commands

# Rule to build all files generated by this target.
CMakeFiles/commands.dir/build: commands
.PHONY : CMakeFiles/commands.dir/build

CMakeFiles/commands.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/commands.dir/cmake_clean.cmake
.PHONY : CMakeFiles/commands.dir/clean

CMakeFiles/commands.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/commands.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/commands.dir/depend
