cmake_minimum_required(VERSION 3.5) # (1) necessary
# ament full documentation can be find here https://docs.ros.org/en/foxy/Guides/Ament-CMake-Documentation.html

# (2) necessary - the argument to project() will be the package name and must be identical to the package name in xml file
project(ceiling_perception)
set(FastRTPS_INCLUDE_DIR /opt/ros/rolling/include)
set(FastRTPS_LIBRARY_RELEASE /opt/ros/rolling/lib/libfastrtps.so)
set(lib_name ${PROJECT_NAME}) # arbitrary

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DDEBUG)
endif()

# (6) Compiler and linker option
#       - ROS2 supports c++14 and c99
#       - To keep the code clean, It is recommended to at least cover the following warning levels
#         For GCC and Clang: -Wall -Wextra -Wpedantic are required and -Wshadow -Werror are advisable (the latter makes warnings errors)
if(NOT CMAKE_C_STANDARD)
    set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 14)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic ) #-Wshadow -Werror
endif()


# (4) There are two ways to link your packages against a new dependency
#     - The first and recommended way is to use the ament macro ament_target_dependencies
#         find_package(Eigen3 REQUIRED)
#         ament_target_dependencies(my_target Eigen3)
#       It includes the necessary headers and libraries and their dependencies to be correctly found by the project.
#       It will also ensure that the include directories of all dependencies are ordered correctly when using overlay workspaces.
#
#     - The second way is to use target_link_libraries
#         find_package(Eigen3 REQUIRED)
#         target_link_libraries(my_target Eigen3::Eigen)
#       This will also include necessary headers, libraries and their dependencies, but in contrast to ament_target_dependencies
#       it might not correctly order the dependencies when using overlay workspaces.
#
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(custom_roi_srv REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

set(dependencies
        rclcpp
        nav2_costmap_2d
        pluginlib
        sensor_msgs
        geometry_msgs
        std_msgs
        custom_roi_srv
        OpenCV
        cv_bridge
        )

# (5) There are two main targets to build:
#   - libraries which are built by add_library
#   - executables which are built by add_executable

#   With the separation of header files and implementation in C/C++,
#   it is not always necessary to add both files as argument to add_library/ add_executable

#   The following best practice is proposed:
#     - if you are building a library, put all headers which should be usable by clients
#       and therefore must be installed into a subdirectory of the include folder named like the package,
#       while all other files (.c/.cpp and header files which should not be exported) are inside the src folder.
#     - only cpp files are explicitly referenced in the call to add_library or add_executable
#     - allow to find headers via
#       target_include_directories(my_target
#         PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#         $<INSTALL_INTERFACE:include>)
#       This adds all files in the folder ${CMAKE_CURRENT_SOURCE_DIR}/include to the public interface during build time
#       and all files in the include folder (relative to ${CMAKE_INSTALL_DIR}) when being installed.
#

include_directories(include)


# === Build ===

add_library(${lib_name} SHARED
        src/ceiling_perception.cpp
        src/overhead_camera_manager.cpp
        src/array_parser.cpp)
ament_target_dependencies(${lib_name} ${dependencies})

# the following three are added to main.cpp be known as an executable and it knows ROS
add_executable(ceiling_perception_start src/main.cpp)
target_link_libraries(ceiling_perception_start ${lib_name})
ament_target_dependencies(ceiling_perception_start ${dependencies})

# === Installation ===



# for the launch file
install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}/
        )
# for the config file to be find in install folder
install(DIRECTORY
        config
        DESTINATION share/${PROJECT_NAME}
        )

# installs the library. Archives and library files will be exported to the lib folder,
# runtime binaries will be installed to the bin folder and the path to installed headers is include
install(TARGETS ceiling_perception_start ${lib_name}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib/${PROJECT_NAME})

# installs the header files which should be available to clients.
install(DIRECTORY include/
        DESTINATION include/
        )

# marks the directory of the exported include directories (this is achieved by INCLUDES DESTINATION in the target install call)
ament_export_include_directories(include)
# marks the location of the installed library (this is done by the HAS_LIBRARY_TARGET argument in the call to ament_export_targets
ament_export_libraries(${lib_name})
# exports dependencies to downstream packages.
# This is necessary so that the user of the library does not have to call find_package for those dependencies, too
ament_export_dependencies(${lib_name} ${dependencies})


# just to clion Knows the files for better interaction
#set(
#        SOURCE_FILES
#        include/ceiling_perception/overhead_camera_manager.h
#        src/overhead_camera_manager.cpp
#        include/ceiling_perception/ceiling_perception.h
#        src/ceiling_perception.cpp
#        include/ceiling_perception/array_parser.h
#        src/array_parser.cpp
#        src/main.cpp
#)

#
# (3) necessary - The project setup is done by ament_package() and this call must occur exactly once per package.
#   ament_package() installs the package.xml, registers the package with the ament index, and installs config (and possibly target) files for CMake
#   so that it can be found by other packages using find_package.
#   Since ament_package() gathers a lot of information from the CMakeLists.txt it should be the last call in your CMakeLists.txt.
#   Although it is possible to follow calls to ament_package() by calls to install functions copying files and directories,
#   it is simpler to just keep ament_package() the last call.
ament_package()
