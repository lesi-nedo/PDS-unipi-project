set(
    FF_RF_CONF_FILE
    "${CMAKE_SOURCE_DIR}/src/fastflow/ff.conf"
    CACHE STRING
    "Path to the FastFlow DecisionForest configuration file"
)

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

set(RESULTS_PATH_EXP2
    "${CMAKE_SOURCE_DIR}/results/fastflow_impl2/"
    CACHE PATH
    "Path to store results for FastFlow implementation 2"
)
set(RESULTS_PATH_EXP1
    "${CMAKE_SOURCE_DIR}/results/fastflow_impl1/"
    CACHE PATH
    "Path to store results for FastFlow implementation 1"
)


configure_file(
    "${CMAKE_SOURCE_DIR}/src/fastflow/ff_impl_config.h.in"
    "${CMAKE_SOURCE_DIR}/src/fastflow/ff_impl_config.h"
)

add_executable(
    ff_impl_exp1
    "${CMAKE_CURRENT_SOURCE_DIR}/main_exp1.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp" 
)

add_executable(
    ff_impl_exp2
    "${CMAKE_CURRENT_SOURCE_DIR}/main_exp2.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp" 
)

target_link_libraries(ff_impl_exp1 PRIVATE TBB::tbb)
target_link_libraries(ff_impl_exp2 PRIVATE TBB::tbb)

target_include_directories(
    ff_impl_exp1
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/"
    "${CMAKE_SOURCE_DIR}/src/utils" 
    "${CMAKE_SOURCE_DIR}/src/"
    "${CMAKE_SOURCE_DIR}/external/fastflow/"
)

target_include_directories(
    ff_impl_exp2
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/"
    "${CMAKE_SOURCE_DIR}/src/utils" 
    "${CMAKE_SOURCE_DIR}/src/"
    "${CMAKE_SOURCE_DIR}/external/fastflow/"
)

add_custom_target(
    run_ff_impl_exp1
    COMMAND $<TARGET_FILE:ff_impl_exp1> ${TRAIN_DATA_PATH} ${TEST_DATA_PATH} >> /dev/null 2>&1
    DEPENDS ff_impl_exp1
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running FastFlow implementation 1 on Susy dataset"
)

add_custom_target(
    run_ff_impl_exp2
    COMMAND $<TARGET_FILE:ff_impl_exp2> ${TRAIN_DATA_PATH} ${TEST_DATA_PATH} >> /dev/null 2>&1
    DEPENDS ff_impl_exp2
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running FastFlow implementation 2 on Susy dataset"
)

add_custom_target(
    run_ff_impl_comparison
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/src/utils/analyze_performance.py"
            "${RESULTS_PATH_EXP2}/performance.csv"
            "${RESULTS_PATH_EXP1}/performance.csv"
    DEPENDS run_ff_impl_exp1 run_ff_impl_exp2
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/"
    COMMENT "Comparing FastFlow implementations 1 and 2"
)