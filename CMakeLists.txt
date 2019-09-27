cmake_minimum_required(VERSION 3.0)
project(CEGO)

# We heavily use C++17, C++17 support is required
set (CMAKE_CXX_STANDARD 17)

# Add the pybind11 stuff
if (NOT CEGO_NO_PYTHON)
  add_subdirectory("${CMAKE_SOURCE_DIR}/externals/pybind11" "pybind11")
endif()

# No more CRT secure warnings...
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

macro(add_nix_libraries target)
    # See https://stackoverflow.com/a/29871891
    # Add DL and pthreads
    FIND_PACKAGE ( Threads REQUIRED )
    find_package(Threads REQUIRED)
    if(THREADS_HAVE_PTHREAD_ARG)
      target_compile_options(${target} PRIVATE "-pthread")
    endif()
    if(CMAKE_THREAD_LIBS_INIT)
      target_link_libraries(${target} PRIVATE "${CMAKE_THREAD_LIBS_INIT}")
    endif()
endmacro()

function(attach_includes target)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch/single_include")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann")
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
  )
  if (NOT MSVC)
    add_nix_libraries(${target})
  endif()
  if (MSVC)
      target_sources(${target} PRIVATE "${CMAKE_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
  endif()
endfunction()

# Always build the interface target 
add_library(CEGO INTERFACE)
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/autodiff")
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/Catch/single_include")
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
target_include_directories(CEGO PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/externals/nlohmann")
set_target_properties(CEGO PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
)

if (NOT CEGO_NO_PYTHON)
  # Build pybind11 python module
  pybind11_add_module(PyCEGO "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx")
  target_compile_definitions(PyCEGO PUBLIC -DPYBIND11)
  attach_includes(PyCEGO)
endif()

if (CEGO_ALL)
  set(APP_SOURCES)

  # Collect all the snippets in the src folder
  file(GLOB_RECURSE snippets "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx")
  message(STATUS "snippets found = ${snippets}")

  foreach (snippet ${snippets})
    get_filename_component(snippet_exe ${snippet} NAME_WE)
    add_executable(${snippet_exe} ${snippet})
    if (NOT CEGO_NO_PYTHON)
      target_link_libraries (${snippet_exe} PRIVATE pybind11::embed)
    endif()
    attach_includes(${snippet_exe})
  endforeach()

endif()