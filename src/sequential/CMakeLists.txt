set(RESULT_PATH "${CMAKE_SOURCE_DIR}/results/sequential")
if(NOT EXISTS "${RESULT_PATH}")
    file(MAKE_DIRECTORY "${RESULT_PATH}")
endif()

set(PATH_FILES "${CMAKE_SOURCE_DIR}/src/data/paths.txt")
if(NOT EXISTS "${PATH_FILES}")
    message(FATAL_ERROR "Paths file not found: ${PATH_FILES}. Please ensure the preprocessing script has created it.")
endif()

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/src/data"
    "${CMAKE_CURRENT_BINARY_DIR}/data"
)

file(STRINGS "${PATH_FILES}" paths_content)
string(REPLACE "\n" ";" paths_list "${paths_content}")
list(GET paths_list 0 TRAIN_DATA_PATH)
list(GET paths_list 1 TEST_DATA_PATH)
if(NOT TRAIN_DATA_PATH OR NOT TEST_DATA_PATH)
    message(FATAL_ERROR "Train or test data paths not found in environment variables")
endif()

message(STATUS "Train data path: ${TRAIN_DATA_PATH}")
message(STATUS "Test data path: ${TEST_DATA_PATH}")

add_executable(
    sequential_impl
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
)

target_include_directories(
    sequential_impl
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/sequential"
    "${CMAKE_SOURCE_DIR}/src/utils"
)

add_custom_target(
    run_sequential
    COMMAND $<TARGET_FILE:sequential_impl> "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}"
    DEPENDS sequential_impl
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Sequential implementation with SUSY dataset..."
)
