cmake_minimum_required(VERSION 3.0)
project(CEGO)

# We heavily use C++11, C++11 support is required
set (CMAKE_CXX_STANDARD 11)

# Add the pybind11 stuff
add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11" "pybind11")

function(attach_includes target)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch/single_include")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann")
endfunction()

# Build pybind11 python module
pybind11_add_module(PyCEGO "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx")
target_compile_definitions(PyCEGO PUBLIC -DPYBIND11)
attach_includes(PyCEGO)

if (CEGO_ALL)
  set(APP_SOURCES)
  if (MSVC)
      list(APPEND APP_SOURCES "${CMAKE_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()

  # Collect all the snippets in the src folder
  file(GLOB_RECURSE snippets "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
  message(STATUS "snippets found = ${snippets}")

  foreach (snippet ${snippets})
    get_filename_component(snippet_exe ${snippet} NAME_WE)
    add_executable(${snippet_exe} ${snippet})
    target_link_libraries (${snippet_exe} PRIVATE pybind11::embed)
    attach_includes(${snippet_exe})
  endforeach()

endif()