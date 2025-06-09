find_package(GTest REQUIRED)

add_executable(test_utils
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/test_utils.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/utils.h"
)

target_include_directories(test_utils
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/utils"
        "${CMAKE_SOURCE_DIR}/include/sequential"
)
target_link_libraries(test_utils GTest::gtest GTest::gtest_main)

add_test(NAME test_utils COMMAND test_utils)
