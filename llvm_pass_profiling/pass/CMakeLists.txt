cmake_minimum_required(VERSION 3.20)
# C is enabled too: LLVM's config-mode dependencies (FindLibEdit) probe with
# check_include_file, which needs a C compiler.
project(LoopStridePass C CXX)

# Point CMake at the LLVM you build against, e.g.
#   cmake -S . -B build -DLLVM_DIR=$(llvm-config --cmakedir)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Building against LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

add_library(LoopStridePass MODULE LoopStridePass.cpp)
target_include_directories(LoopStridePass PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(LoopStridePass PRIVATE ${LLVM_DEFINITIONS_LIST})
target_compile_features(LoopStridePass PRIVATE cxx_std_17)

# LLVM is built without RTTI; the plugin must match.
target_compile_options(LoopStridePass PRIVATE -fno-rtti)

# The plugin resolves LLVM symbols from the host binary (opt/clang) at load
# time, so leave them undefined at link time on macOS.
if(APPLE)
  set_target_properties(LoopStridePass PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
endif()
