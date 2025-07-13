set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(IRIS_DIR "${DATA_DIR}/iris")
set(IRIS_DATASET "${IRIS_DIR}/iris.data")

message(STATUS "Checking required Python packages...")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "Checking for Iris dataset...")
if(NOT EXISTS "${IRIS_DATASET}")
    if(NOT EXISTS "${DATA_DIR}")
        message(STATUS "Creating data directory at ${DATA_DIR}...")
        file(MAKE_DIRECTORY "${DATA_DIR}")
    endif()
    message(STATUS "Iris dataset not found at ${IRIS_DATASET}. Downloading...")
    file(DOWNLOAD
        "https://archive.ics.uci.edu/static/public/53/iris.zip"
        "${IRIS_DIR}/iris.zip"
        STATUS download_status
        LOG download_log
    )

    list (GET download_status 0 download_result)
    if(download_result EQUAL 0)
        message(STATUS "Iris dataset downloaded successfully.")
        message(STATUS "Unzipping iris.zip...")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${IRIS_DIR}/iris.zip"
            WORKING_DIRECTORY "${IRIS_DIR}"
            RESULT_VARIABLE unzip_result
        )
        if(unzip_result EQUAL 0)
            message(STATUS "Iris dataset unzipped successfully.")
            file(REMOVE "${IRIS_DIR}/iris.zip")
            message(STATUS "Removed iris.zip after extraction.")
        else()
            message(FATAL_ERROR "Failed to unzip iris.zip: ${unzip_result}")
        endif()
    else()
        message(FATAL_ERROR "Failed to download iris dataset: ${download_log}") 
    endif()
else()
    message(STATUS "Iris dataset already exists at ${IRIS_DATASET}.")
endif()

# Execute the preprocessing script
message(STATUS "Running data preprocessing...")
execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/preprocess.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RESULT_VARIABLE preprocess_result
    OUTPUT_VARIABLE preprocess_output
    ERROR_VARIABLE preprocess_error
)

if(preprocess_result EQUAL 0)
    message(STATUS "Data preprocessing completed successfully.")
    if(preprocess_output)
        message(STATUS "Output: ${preprocess_output}")
    endif()
else()
    message(FATAL_ERROR "Data preprocessing failed: ${preprocess_error}")
endif()

set(PATHS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/iris_paths.txt")

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
file(REMOVE "${PATHS_FILE}")

add_executable(sq_random_forest
    "${CMAKE_CURRENT_SOURCE_DIR}/sq_random_forest.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
    )


target_include_directories(sq_random_forest
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/utils"
        "${CMAKE_SOURCE_DIR}/include/sequential"
        "${CMAKE_SOURCE_DIR}/include"
)

add_custom_target(
    run_sq_random_forest
    COMMAND $<TARGET_FILE:sq_random_forest> "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}"
    DEPENDS sq_random_forest
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Running Sequential Random Forest on Iris dataset"
    )

add_custom_target(
    run_sequential_validation
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/validate.py" "${TRAIN_DATA_PATH}" "${TEST_DATA_PATH}"
    DEPENDS run_sq_random_forest
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )