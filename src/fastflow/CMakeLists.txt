
set(PATHS_FILE "${CMAKE_SOURCE_DIR}/src/data/paths.txt")

set(FF_RF_CONF_FILE "${CMAKE_SOURCE_DIR}/src/fastflow/default.conf" CACHE STRING "The path to the configuration file for FastFlow implementation")

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/src/data"
    "${CMAKE_CURRENT_BINARY_DIR}/data"
)

if(NOT EXISTS "${PATHS_FILE}")
    message(FATAL_ERROR "Paths file not found: ${PATHS_FILE}. Please ensure the preprocessing script has created it.")
endif()

file(STRINGS "${PATHS_FILE}" paths_content)
string(REPLACE "\n" ";" paths_list "${paths_content}")
list(GET paths_list 0 TRAIN_DATA_PATH)
list(GET paths_list 1 TEST_DATA_PATH)

if(NOT TRAIN_DATA_PATH OR NOT TEST_DATA_PATH)
    message(FATAL_ERROR "Train or test data paths not found in environment variables")
endif()

message(STATUS "Train data path: ${TRAIN_DATA_PATH}")
message(STATUS "Test data path: ${TEST_DATA_PATH}")

message(STATUS "Creating config.h file for FastFlow implementation...")

message(STATUS "Configuarion file for FastFlow implementation: ${FF_RF_CONF_FILE}")

if(NOT EXISTS "${FF_RF_CONF_FILE}")
    message(FATAL_ERROR "Configuration file not found: ${FF_RF_CONF_FILE}. Please provide a valid configuration file.")
endif()

file(STRINGS "${FF_RF_CONF_FILE}" CONF_LINES)

foreach(line IN LISTS CONF_LINES)
    if(line MATCHES "^\\s*#")  # Skip comments
        continue()
    endif()
    if(line MATCHES "^\\s*$")  # Skip empty lines
        continue()
    endif()

    # Use REGEX MATCH for cleaner and more robust parsing.
    # This handles potential whitespace and correctly extracts key and value.
    if(line MATCHES "^[ ]*([a-zA-Z0-9_]+)[ ]*=[ ]*(.+)$")
       
        #   # Remove comments
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




configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ff_impl_config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/ff_impl_config.h"
)

add_executable(
    ff_impl_exp1
    "${CMAKE_CURRENT_SOURCE_DIR}/main_exp1.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp" 
)

target_include_directories(
    ff_impl_exp1
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/sequential"
    "${CMAKE_SOURCE_DIR}/src/utils" 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/external/fastflow/"
)