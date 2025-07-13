set(
    OMP_RF_CONF_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/omp.conf"
    CACHE STRING
    "Path to the OpenMP DecisionForest configuration file"
)


if(NOT EXISTS "${OMP_RF_CONF_FILE}")
    message(FATAL_ERROR "Configuration file not found: ${OMP_RF_CONF_FILE}. Please provide a valid configuration file.")
endif()

file(STRINGS "${OMP_RF_CONF_FILE}" CONF_LINES)

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
endforeach()

set(RESULTS_PATH_OMP
    "${CMAKE_SOURCE_DIR}/results/openmp/"
    CACHE PATH
    "Path to store results for OpenMP implementation"
)

if(NOT EXISTS "${RESULTS_PATH_OMP}")
    file(MAKE_DIRECTORY "${RESULTS_PATH_OMP}")
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/omp_impl_config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/omp_impl_config.h"
)

add_executable(
    omp_impl
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
)

target_include_directories(
    omp_impl
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/utils"
    "${CMAKE_SOURCE_DIR}/include/"
    "${CMAKE_SOURCE_DIR}/src"
)

add_custom_target(
    run_omp_impl
    COMMAND $<TARGET_FILE:omp_impl> ${TRAIN_DATA_PATH} ${TEST_DATA_PATH}
    DEPENDS omp_impl
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running OpenMP implementation of DecisionForest"
)

