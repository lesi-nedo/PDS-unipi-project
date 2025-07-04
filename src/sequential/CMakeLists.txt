set(
    SEQ_RESULTS_PATH
    "${CMAKE_SOURCE_DIR}/results/sequential_impl/"
    CACHE PATH
    "Path to store results for Sequential implementation"
)


configure_file(
  "${CMAKE_SOURCE_DIR}/src/sequential/seq_impl_config.h.in"
  "${CMAKE_SOURCE_DIR}/src/sequential/seq_impl_config.h"
)

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
    "${CMAKE_SOURCE_DIR}/src/"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(
    run_sequential
    COMMAND $<TARGET_FILE:sequential_impl> "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}"
    DEPENDS sequential_impl
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Sequential implementation with SUSY dataset..."
)
