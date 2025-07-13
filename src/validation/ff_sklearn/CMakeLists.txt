set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(SUSY_DIR "${DATA_DIR}/susy")
set(PATHS_FILE "${CMAKE_SOURCE_DIR}/src/data/susy_paths.txt")

add_subdirectory(
    "${CMAKE_SOURCE_DIR}/src/data"
    "${CMAKE_CURRENT_BINARY_DIR}/src/fastflow/"
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

add_executable(
    local_ff_impl_exp2
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
)

target_include_directories(
    local_ff_impl_exp2
    PRIVATE
    "${CMAKE_SOURCE_DIR}/include/sequential"
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src/utils"
    "${CMAKE_SOURCE_DIR}/src/"
    "${CMAKE_SOURCE_DIR}/src/fastflow"
    "${CMAKE_SOURCE_DIR}/external/fastflow"
)
# Ask an input to the user to choose the implementation

add_custom_target(
    run_local_ff_impl_exp2
    COMMAND $<TARGET_FILE:local_ff_impl_exp2> "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}"
    DEPENDS local_ff_impl_exp2
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running FastFlow Random Forest on SUSY dataset"
)

add_custom_target(
    run_ff_impl_validation
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/src/validation/ff_sklearn/validate.py"
            "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}" "${CMAKE_SOURCE_DIR}/results/validation/ff_sklearn/"
    DEPENDS run_local_ff_impl_exp2
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running FastFlow Validation on SUSY dataset"
)