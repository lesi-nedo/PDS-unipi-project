cmake_minimum_required(VERSION 3.20)
project(RandomForsetParallel VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    add_definitions(-DNDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -ffast-math -march=native")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_FASTFLOW "Build FastFlow implementation" ON)
option(BUILD_OPENMP "Build OpenMP implementation" ON)
option(BUILD_MPI "Build MPI implementation with FastFlow" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_VALSEQ "Build sequential Random Forset and Validate against Scikit-learn One" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# add_subdirectory(utils)

find_package(Threads REQUIRED)
if(NOT THREADS_FOUND)
    message(FATAL_ERROR "Pthreads not found. Please ensure you have pthreads installed.")
endif()

if(BUILD_RFCOMPAR)
    message(STATUS "Building sequential Random Forset Comparison with Scikit-learn")
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python3 not found. Please ensure you have Python3 installed.")
    endif()
    
    # Check for NumPy
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
        RESULT_VARIABLE NUMPY_RESULT
        OUTPUT_VARIABLE NUMPY_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NUMPY_RESULT EQUAL 0)
        message(STATUS "NumPy found: ${NUMPY_VERSION}")
    else()
        message(FATAL_ERROR "NumPy not found. Please install it with: pip install numpy")
    endif()
    
    # Check for Scikit-learn
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sklearn; print(sklearn.__version__)"
        RESULT_VARIABLE SKLEARN_RESULT
        OUTPUT_VARIABLE SKLEARN_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(SKLEARN_RESULT EQUAL 0)
        message(STATUS "Scikit-learn found: ${SKLEARN_VERSION}")
    else()
        message(STATUS "In Ubuntu, you may need to install the `python3-sklearn` package by running: sudo apt-get install python3-sklearn python3-sklearn-lib python-sklearn-doc")
        message(FATAL_ERROR "Scikit-learn not found. Please install it with: pip install scikit-learn")
    endif()
    add_subdirectory(src/validation/seq_sklearn)
endif()

if(BUILD_OPENMP)
    find_package(OpenMP REQUIRED)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "OpenMP not found. Please ensure you have OpenMP installed.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not requested, skipping OpenMP configuration.")
endif()

if(BUILD_MPI)
    find_package(MPI REQUIRED)
    if(NOT MPI_CXX_FOUND)
        message(FATAL_ERROR "MPI not found. Please ensure you have MPI installed.")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    message(STATUS "MPI found: ${MPI_CXX_COMPILE_FLAGS}")
else()
    message(STATUS "MPI not requested, skipping MPI configuration.")
endif()


if(BUILD_FASTFLOW)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow/CMakeLists.txt")
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(HWLOC hwloc)
        endif()

        if(NOT HWLOC_FOUND)
            find_path(HWLOC_INCLUDE_DIR hwloc.h
                HINTS /usr/include /usr/local/include
                PATH_SUFFIXES hwloc
            )
            find_library(HWLOC_LIBRARY NAMES hwloc
                HINTS /usr/lib /usr/local/lib
            )
            if(HWLOC_INCLUDE_DIR AND HWLOC_LIBRARY)
                set(HWLOC_FOUND TRUE)
                set(HWLOC_LIBRARIES ${HWLOC_LIBRARY})
                set(HWLOC_INCLUDE_DIRS ${HWLOC_INCLUDE_DIR})
                message(STATUS "Found hwloc: ${HWLOC_INCLUDE_DIR}, ${HWLOC_LIBRARY}")
            else()
                message(STATUS "Package needed for FastFlow not found.")
                message(FATAL_ERROR "hwloc not found. Please install hwloc or set HWLOC_INCLUDE_DIR and HWLOC_LIBRARY manually.")
            endif()
        else()
            message(STATUS "hwloc found via pkg-config: ${HWLOC_INCLUDE_DIR}, ${HWLOC_LIBRARY}")
            include(FetchContent)
            FetchContent_Declare(
                fastflow
                GIT_REPOSITORY https://github.com/fastflow/fastflow.git
                GIT_TAG        master
                SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow"
            )
            message(STATUS "Fetching FastFlow from repository...")
            FetchContent_Populate(fastflow)
            message(STATUS "Running CMake configuration for FastFlow...")
            set(FASTFLOW_CMAKELISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow/CMakeLists.txt")
            if(EXISTS "${FASTFLOW_CMAKELISTS}")
                file(READ "${FASTFLOW_CMAKELISTS}" FASTFLOW_CMAKE_CONTENT)
                string(REPLACE "CMAKE_HAVE_PTHREAD_H" "THREADS_FOUND" FASTFLOW_CMAKE_CONTENT "${FASTFLOW_CMAKE_CONTENT}")
                file(WRITE "${FASTFLOW_CMAKELISTS}" "${FASTFLOW_CMAKE_CONTENT}")
            else()
                message(FATAL_ERROR "CMakeLists.txt for FastFlow not found at ${FASTFLOW_CMAKELISTS}.")
            endif()
            message(STATUS "Running CMake configuration for FastFlow...")
            FetchContent_MakeAvailable(fastflow)
            message(STATUS "FastFlow library fetched and added.")
            message(STATUS "FastFlow version: ${fastflow_VERSION}")
            message(STATUS "Running `mapping_string.sh` script to update configurations.")
            execute_process(
                COMMAND  bash -c "echo 'y' | \"${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow/ff/mapping_string.sh\""
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow/ff"
                RESULT_VARIABLE result_mapping
                OUTPUT_VARIABLE output_mapping
                ERROR_VARIABLE error_mapping
            )
            if(result_mapping EQUAL 0)
                message(STATUS "Mapping string script executed successfully: ${output_mapping}")
            else()
                message(WARNING "Mapping string script failed: ${error_mapping}")
            endif()


        endif()
    endif()
    set(FF_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fastflow/ff")

    add_library(fastflow INTERFACE)
    target_include_directories(fastflow INTERFACE
        "${FF_INCLUDE_DIR}"
    )
    target_link_libraries(fastflow INTERFACE
        Threads::Threads)
    
    set_target_properties(fastflow PROPERTIES EXPORT_NAME ff)
    add_library(ff::ff ALIAS fastflow)
    message(STATUS "FastFlow library configured successfully.")
    
else()
    message(STATUS "FastFlow not requested, skipping FastFlow configuration.")
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(src/tests)
else()
    message(STATUS "Tests not requested, skipping tests configuration.")
endif()

if(BUILD_VALSEQ)
    message(STATUS "Building sequential Random Forset and validation against Scikit-learn")
    add_subdirectory(src/validation/seq_sklearn)
endif()    

message (STATUS "-- Configuration Summary --")
message (STATUS "CMake version: ${CMAKE_VERSION}")
message (STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")
message (STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message (STATUS "C++ standard required: ${CMAKE_CXX_STANDARD_REQUIRED}")
message (STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message (STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message (STATUS "C++ linker flags: ${CMAKE_EXE_LINKER_FLAGS}")
message (STATUS "Pthreads found: ${THREADS_FOUND}")
message (STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
message (STATUS "MPI found: ${MPI_CXX_FOUND}")
message (STATUS "FastFlow found: ${BUILD_FASTFLOW}")
message (STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")
message (STATUS "MPI compile flags: ${MPI_CXX_COMPILE_FLAGS}")
message (STATUS "MPI link flags: ${MPI_CXX_LINK_FLAGS}")
message (STATUS "Utilities built: ${BUILD_UTILS}")
message (STATUS "Tests built: ${BUILD_TESTS}")
message (STATUS "FastFlow version: ${fastflow_VERSION}")
message(STATUS "-- End of Configuration Summary --")
