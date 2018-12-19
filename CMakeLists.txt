cmake_minimum_required(VERSION 3.9)
project(keigen)

set(CMAKE_CXX_STANDARD 11)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(.)

add_executable(r_traits traits.cpp)
add_executable(r_boost boost.cpp)
add_executable(r_kvm kvm.cpp)
add_executable(r_kq kq.cpp Eigen)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs core)

# Link against LLVM libraries
target_link_libraries(r_kvm ${llvm_libs})
#target_link_libraries(r_kvm Eigen3::Eigen)
#target_link_libraries(r_kq Eigen3::Eigen)