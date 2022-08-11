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
include CMakeFiles/ffnn_gpu_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ffnn_gpu_lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ffnn_gpu_lib.dir/flags.make

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o: applications/ffnn-gpu/ffnn_lib_init.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_lib_init.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_lib_init.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_lib_init.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o: applications/ffnn-gpu/ffnn_activation_mult.cu
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o.d -x cu -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_activation_mult.cu -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o: applications/ffnn-gpu/ffnn_add.cu
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o.d -x cu -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_add.cu -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o: applications/ffnn-gpu/ffnn_mult.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_mult.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_mult.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_mult.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o: applications/ffnn-gpu/ffnn_uniform_data.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_data.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_data.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_data.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o: applications/ffnn-gpu/ffnn_uniform_weights.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_weights.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_weights.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_weights.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o: applications/ffnn-gpu/ffnn_types.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_types.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_types.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_types.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o: applications/ffnn-gpu/ffnn_weighted_sum.cu
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CUDA object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o.d -x cu -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_weighted_sum.cu -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o: applications/ffnn-gpu/ffnn_matrix_hadamard.cu
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CUDA object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o.d -x cu -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_matrix_hadamard.cu -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o: applications/ffnn-gpu/ffnn_back_mult.cu
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CUDA object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o.d -x cu -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_back_mult.cu -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o: applications/ffnn-gpu/ffnn_uniform_sparse_data.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.s

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o: CMakeFiles/ffnn_gpu_lib.dir/flags.make
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o: applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc
CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o: CMakeFiles/ffnn_gpu_lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o -MF CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o.d -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o -c /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.i"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc > CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.i

CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.s"
	/opt/intel/oneapi/compiler/latest/linux/bin/icpx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dimitrije/bbts/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc -o CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.s

# Object files for target ffnn_gpu_lib
ffnn_gpu_lib_OBJECTS = \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o" \
"CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o"

# External object files for target ffnn_gpu_lib
ffnn_gpu_lib_EXTERNAL_OBJECTS =

libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_lib_init.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_activation_mult.cu.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_add.cu.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_mult.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_data.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_weights.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_types.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum.cu.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_matrix_hadamard.cu.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_back_mult.cu.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_uniform_sparse_data.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/applications/ffnn-gpu/ffnn_weighted_sum_sparse_dense.cc.o
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/build.make
libraries/libffnn_gpu_lib.so: /usr/local/cuda/lib64/libcurand.so
libraries/libffnn_gpu_lib.so: CMakeFiles/ffnn_gpu_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dimitrije/bbts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX shared library libraries/libffnn_gpu_lib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ffnn_gpu_lib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ffnn_gpu_lib.dir/build: libraries/libffnn_gpu_lib.so
.PHONY : CMakeFiles/ffnn_gpu_lib.dir/build

CMakeFiles/ffnn_gpu_lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ffnn_gpu_lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ffnn_gpu_lib.dir/clean

CMakeFiles/ffnn_gpu_lib.dir/depend:
	cd /home/dimitrije/bbts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts /home/dimitrije/bbts/CMakeFiles/ffnn_gpu_lib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ffnn_gpu_lib.dir/depend

