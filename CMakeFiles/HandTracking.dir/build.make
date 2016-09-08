# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/ray/Documents/FinalProject/HandTracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ray/Documents/FinalProject/HandTracking

# Include any dependencies generated for this target.
include CMakeFiles/HandTracking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/HandTracking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/HandTracking.dir/flags.make

CMakeFiles/HandTracking.dir/HandTracking.cpp.o: CMakeFiles/HandTracking.dir/flags.make
CMakeFiles/HandTracking.dir/HandTracking.cpp.o: HandTracking.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ray/Documents/FinalProject/HandTracking/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/HandTracking.dir/HandTracking.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/HandTracking.dir/HandTracking.cpp.o -c /home/ray/Documents/FinalProject/HandTracking/HandTracking.cpp

CMakeFiles/HandTracking.dir/HandTracking.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/HandTracking.dir/HandTracking.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ray/Documents/FinalProject/HandTracking/HandTracking.cpp > CMakeFiles/HandTracking.dir/HandTracking.cpp.i

CMakeFiles/HandTracking.dir/HandTracking.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/HandTracking.dir/HandTracking.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ray/Documents/FinalProject/HandTracking/HandTracking.cpp -o CMakeFiles/HandTracking.dir/HandTracking.cpp.s

CMakeFiles/HandTracking.dir/HandTracking.cpp.o.requires:
.PHONY : CMakeFiles/HandTracking.dir/HandTracking.cpp.o.requires

CMakeFiles/HandTracking.dir/HandTracking.cpp.o.provides: CMakeFiles/HandTracking.dir/HandTracking.cpp.o.requires
	$(MAKE) -f CMakeFiles/HandTracking.dir/build.make CMakeFiles/HandTracking.dir/HandTracking.cpp.o.provides.build
.PHONY : CMakeFiles/HandTracking.dir/HandTracking.cpp.o.provides

CMakeFiles/HandTracking.dir/HandTracking.cpp.o.provides.build: CMakeFiles/HandTracking.dir/HandTracking.cpp.o

# Object files for target HandTracking
HandTracking_OBJECTS = \
"CMakeFiles/HandTracking.dir/HandTracking.cpp.o"

# External object files for target HandTracking
HandTracking_EXTERNAL_OBJECTS =

HandTracking: CMakeFiles/HandTracking.dir/HandTracking.cpp.o
HandTracking: CMakeFiles/HandTracking.dir/build.make
HandTracking: /usr/local/lib/libopencv_videostab.so.3.0.0
HandTracking: /usr/local/lib/libopencv_videoio.so.3.0.0
HandTracking: /usr/local/lib/libopencv_video.so.3.0.0
HandTracking: /usr/local/lib/libopencv_superres.so.3.0.0
HandTracking: /usr/local/lib/libopencv_stitching.so.3.0.0
HandTracking: /usr/local/lib/libopencv_shape.so.3.0.0
HandTracking: /usr/local/lib/libopencv_photo.so.3.0.0
HandTracking: /usr/local/lib/libopencv_objdetect.so.3.0.0
HandTracking: /usr/local/lib/libopencv_ml.so.3.0.0
HandTracking: /usr/local/lib/libopencv_imgproc.so.3.0.0
HandTracking: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
HandTracking: /usr/local/lib/libopencv_highgui.so.3.0.0
HandTracking: /usr/local/lib/libopencv_hal.a
HandTracking: /usr/local/lib/libopencv_flann.so.3.0.0
HandTracking: /usr/local/lib/libopencv_features2d.so.3.0.0
HandTracking: /usr/local/lib/libopencv_core.so.3.0.0
HandTracking: /usr/local/lib/libopencv_calib3d.so.3.0.0
HandTracking: /usr/local/lib/libopencv_features2d.so.3.0.0
HandTracking: /usr/local/lib/libopencv_ml.so.3.0.0
HandTracking: /usr/local/lib/libopencv_highgui.so.3.0.0
HandTracking: /usr/local/lib/libopencv_videoio.so.3.0.0
HandTracking: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
HandTracking: /usr/local/lib/libopencv_flann.so.3.0.0
HandTracking: /usr/local/lib/libopencv_video.so.3.0.0
HandTracking: /usr/local/lib/libopencv_imgproc.so.3.0.0
HandTracking: /usr/local/lib/libopencv_core.so.3.0.0
HandTracking: /usr/local/lib/libopencv_hal.a
HandTracking: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
HandTracking: CMakeFiles/HandTracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable HandTracking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/HandTracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/HandTracking.dir/build: HandTracking
.PHONY : CMakeFiles/HandTracking.dir/build

CMakeFiles/HandTracking.dir/requires: CMakeFiles/HandTracking.dir/HandTracking.cpp.o.requires
.PHONY : CMakeFiles/HandTracking.dir/requires

CMakeFiles/HandTracking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/HandTracking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/HandTracking.dir/clean

CMakeFiles/HandTracking.dir/depend:
	cd /home/ray/Documents/FinalProject/HandTracking && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ray/Documents/FinalProject/HandTracking /home/ray/Documents/FinalProject/HandTracking /home/ray/Documents/FinalProject/HandTracking /home/ray/Documents/FinalProject/HandTracking /home/ray/Documents/FinalProject/HandTracking/CMakeFiles/HandTracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/HandTracking.dir/depend

