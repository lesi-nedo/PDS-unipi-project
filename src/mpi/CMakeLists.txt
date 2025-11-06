set(
    MPI_RF_CONF_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/mpi.conf"
    CACHE STRING
    "Path to the MPI DecisionForest configuration file"
)

if(NOT EXISTS "${MPI_RF_CONF_FILE}")
    message(FATAL_ERROR "Configuration file not found: ${MPI_RF_CONF_FILE}. Please provide a valid configuration file.")
endif()

file(STRINGS "${MPI_RF_CONF_FILE}" CONF_LINES)

foreach(line IN LISTS CONF_LINES)
   
    if(line MATCHES "^\\s*#+")  # Skip comments
        continue()
    endif()
    if(line MATCHES "^\\s*$")  # Skip empty lines
        continue()
    endif()
    # Use REGEX MATCH for cleaner and more robust parsing.
    # This handles potential whitespace and correctly extracts key and value.
    if(line MATCHES "^[ ]*([a-zA-Z0-9_]+)[ ]*=[ ]*(.+)$")
        
        string(STRIP "${CMAKE_MATCH_2}" value)
        string(REPLACE "\"" "" value "${value}")
        string(TOUPPER "${CMAKE_MATCH_1}" key_upper)

        string(REGEX REPLACE "[ ]*#+.*$" "" value "${value}")
        
        message(STATUS "Setting configuration: ${key_upper} = ${value}")
        set("${key_upper}" "${value}")
        
    else()
        message(FATAL_ERROR "Could not parse configuration line: ${line}")
    endif()
    
endforeach(line IN LISTS CONF_LINES)

set(RESULTS_PATH_MPI
    "${CMAKE_SOURCE_DIR}/results/mpi/"
    CACHE PATH
    "Path to store results for MPI implementation"
)

if(NOT EXISTS "${RESULTS_PATH_MPI}")
    file(MAKE_DIRECTORY "${RESULTS_PATH_MPI}")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mpi_impl_config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/mpi_impl_config.h"
)


add_executable(
    mpi_impl
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
)

target_include_directories(
    mpi_impl
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/utils"
    "${CMAKE_SOURCE_DIR}/src/fastflow"
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/external/fastflow"
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(
    mpi_impl
    PRIVATE
    MPI::MPI_CXX
)