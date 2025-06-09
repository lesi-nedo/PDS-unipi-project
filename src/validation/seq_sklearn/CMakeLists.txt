set(DATA_DIR "${CMAKE_SOURCE_DIR}/data")
set(IRIS_DIR "${DATA_DIR}/iris")
set(IRIS_DATASET "${IRIS_DIR}/iris.data")

if(NOT EXISTS "${IRIS_DATASET)}")
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
