# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/poetry/SRMM/compressai/sadl_codec

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/poetry/SRMM/build

# Include any dependencies generated for this target.
include CMakeFiles/encoder_sadl_int16_generic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/encoder_sadl_int16_generic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/encoder_sadl_int16_generic.dir/flags.make

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o: CMakeFiles/encoder_sadl_int16_generic.dir/flags.make
CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o: /home/poetry/SRMM/compressai/sadl_codec/encoder_int16.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/poetry/SRMM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o -c /home/poetry/SRMM/compressai/sadl_codec/encoder_int16.cpp

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/poetry/SRMM/compressai/sadl_codec/encoder_int16.cpp > CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.i

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/poetry/SRMM/compressai/sadl_codec/encoder_int16.cpp -o CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.s

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.requires:

.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.requires

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.provides: CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.requires
	$(MAKE) -f CMakeFiles/encoder_sadl_int16_generic.dir/build.make CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.provides.build
.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.provides

CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.provides.build: CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o


CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o: CMakeFiles/encoder_sadl_int16_generic.dir/flags.make
CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o: /home/poetry/SRMM/compressai/sadl_codec/range_coder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/poetry/SRMM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o -c /home/poetry/SRMM/compressai/sadl_codec/range_coder.cpp

CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/poetry/SRMM/compressai/sadl_codec/range_coder.cpp > CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.i

CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/poetry/SRMM/compressai/sadl_codec/range_coder.cpp -o CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.s

CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.requires:

.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.requires

CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.provides: CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.requires
	$(MAKE) -f CMakeFiles/encoder_sadl_int16_generic.dir/build.make CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.provides.build
.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.provides

CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.provides.build: CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o


CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o: CMakeFiles/encoder_sadl_int16_generic.dir/flags.make
CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o: /home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/poetry/SRMM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o -c /home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp

CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp > CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.i

CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp -o CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.s

CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.requires:

.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.requires

CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.provides: CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.requires
	$(MAKE) -f CMakeFiles/encoder_sadl_int16_generic.dir/build.make CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.provides.build
.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.provides

CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.provides.build: CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o


# Object files for target encoder_sadl_int16_generic
encoder_sadl_int16_generic_OBJECTS = \
"CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o" \
"CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o" \
"CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o"

# External object files for target encoder_sadl_int16_generic
encoder_sadl_int16_generic_EXTERNAL_OBJECTS =

encoder_sadl_int16_generic: CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o
encoder_sadl_int16_generic: CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o
encoder_sadl_int16_generic: CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o
encoder_sadl_int16_generic: CMakeFiles/encoder_sadl_int16_generic.dir/build.make
encoder_sadl_int16_generic: CMakeFiles/encoder_sadl_int16_generic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/poetry/SRMM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable encoder_sadl_int16_generic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/encoder_sadl_int16_generic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/encoder_sadl_int16_generic.dir/build: encoder_sadl_int16_generic

.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/build

CMakeFiles/encoder_sadl_int16_generic.dir/requires: CMakeFiles/encoder_sadl_int16_generic.dir/encoder_int16.cpp.o.requires
CMakeFiles/encoder_sadl_int16_generic.dir/requires: CMakeFiles/encoder_sadl_int16_generic.dir/range_coder.cpp.o.requires
CMakeFiles/encoder_sadl_int16_generic.dir/requires: CMakeFiles/encoder_sadl_int16_generic.dir/home/poetry/SRMM/third_party/range_coder/range_coder_impl.cpp.o.requires

.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/requires

CMakeFiles/encoder_sadl_int16_generic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/encoder_sadl_int16_generic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/clean

CMakeFiles/encoder_sadl_int16_generic.dir/depend:
	cd /home/poetry/SRMM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/poetry/SRMM/compressai/sadl_codec /home/poetry/SRMM/compressai/sadl_codec /home/poetry/SRMM/build /home/poetry/SRMM/build /home/poetry/SRMM/build/CMakeFiles/encoder_sadl_int16_generic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/encoder_sadl_int16_generic.dir/depend

