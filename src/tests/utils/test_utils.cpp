#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>

#include "utils.h"

namespace fs = std::filesystem;

namespace {
    // Helper function to create a temporary CSV file
    fs::path createTempCSV(const std::string_view content, const std::string_view filename) {
        fs::path tempFile = fs::temp_directory_path() / filename.data();
        std::ofstream outFile(tempFile);
        outFile << content;
        outFile.close();
        return tempFile;
    }
}

// Test fixture for utility functions
class UtilsTest : public ::testing::Test {
    std::vector<std::string_view> all_temp_files = {
        "test_basic.csv",
        "test_header.csv", 
        "test_empty_row.csv",
        "test_header_empty.csv",
        "test_mixed_types.csv",
        "test_string_labels.csv",
        "test_invalid.csv",
        "test_single_row.csv"
    };

protected:
    void SetUp() override {
        // Basic CSV without header
        basicCSV = createTempCSV("1,2,3\n4,5,6\n7,8,9", all_temp_files[0]);
        
        // CSV with header
        headerCSV = createTempCSV("col1,col2,col3\n1,2,3\n4,5,6\n7,8,9", all_temp_files[1]);
        
        // CSV with invalid/empty rows
        emptyRowCSV = createTempCSV("1,2,3\n4,5,6\n\n7,8,9", all_temp_files[2]);
        
        // CSV with header and empty rows
        headerEmptyCSV = createTempCSV("col1,col2,col3\n1,2,3\n\n4,5,6", all_temp_files[3]);
        
        // CSV with mixed numeric data
        mixedCSV = createTempCSV("1.5,2.7,3\n4.1,5.3,6\n7.9,8.2,9", all_temp_files[4]);
        
        // CSV with string labels
        stringLabelCSV = createTempCSV("1,2,A\n3,4,B\n5,6,C", all_temp_files[5]);
        
        // Invalid CSV (all NaN)
        invalidCSV = createTempCSV("NaN,NaN,NaN\nN/A,null,NULL", all_temp_files[6]);
        
        // Single row CSV
        singleRowCSV = createTempCSV("1,2,3", all_temp_files[7]);
    }

    void TearDown() override {
        for (const auto& file : all_temp_files) {
            fs::path tempFile = fs::temp_directory_path() / file.data();
            if (fs::exists(tempFile)) {
                fs::remove(tempFile);
            }
        }
    }

    fs::path basicCSV;
    fs::path headerCSV; 
    fs::path emptyRowCSV;
    fs::path headerEmptyCSV;
    fs::path mixedCSV;
    fs::path stringLabelCSV;
    fs::path invalidCSV;
    fs::path singleRowCSV;
};

// TEST_F(UtilsTest, LoadBasicCSVIntInt) {
//     auto [features, labels] = loadCSVToMarray<int, int>(basicCSV.string(), ',');
    
//     ASSERT_EQ(features.shape(0), 3);  // 3 rows
//     ASSERT_EQ(features.shape(1), 2);  // 2 feature columns
//     ASSERT_EQ(labels.shape(0), 3);    // 3 rows
//     ASSERT_EQ(labels.shape(1), 1);    // 1 label column
    
//     // Check feature values
//     EXPECT_EQ(features(0, 0), 1);
//     EXPECT_EQ(features(0, 1), 2);
//     EXPECT_EQ(features(1, 0), 4);
//     EXPECT_EQ(features(1, 1), 5);
//     EXPECT_EQ(features(2, 0), 7);
//     EXPECT_EQ(features(2, 1), 8);
    
//     // Check label values (last column)
//     EXPECT_EQ(labels(0, 0), 3);
//     EXPECT_EQ(labels(1, 0), 6);
//     EXPECT_EQ(labels(2, 0), 9);
// }

// TEST_F(UtilsTest, LoadCSVWithHeader) {
//     auto [features, labels] = loadCSVToMarray<double, double>(headerCSV.string(), ',');
    
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
//     ASSERT_EQ(labels.shape(0), 3);
    
//     EXPECT_DOUBLE_EQ(features(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(features(0, 1), 2.0);
//     EXPECT_DOUBLE_EQ(labels(0, 0), 3.0);
// }

// TEST_F(UtilsTest, LoadCSVWithEmptyRows) {
//     auto [features, labels] = loadCSVToMarray<int, int>(emptyRowCSV.string(), ',');
    
//     // Should skip empty rows, leaving 3 valid rows
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
//     ASSERT_EQ(labels.shape(0), 3);
    
//     EXPECT_EQ(features(0, 0), 1);
//     EXPECT_EQ(features(1, 0), 4);
//     EXPECT_EQ(features(2, 0), 7);
// }

// TEST_F(UtilsTest, LoadCSVMixedNumericTypes) {
//     auto [features, labels] = loadCSVToMarray<double, int>(mixedCSV.string(), ',');
    
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
    
//     EXPECT_DOUBLE_EQ(features(0, 0), 1.5);
//     EXPECT_DOUBLE_EQ(features(0, 1), 2.7);
//     EXPECT_EQ(labels(0, 0), 3);
// }

// TEST_F(UtilsTest, LoadCSVWithStringLabels) {
//     auto [features, labels] = loadCSVToMarray<int, std::string>(stringLabelCSV.string(), ',');
    
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
//     ASSERT_EQ(labels.shape(0), 3);
    
//     EXPECT_EQ(features(0, 0), 1);
//     EXPECT_EQ(features(0, 1), 2);
//     EXPECT_EQ(labels(0, 0), "A");
//     EXPECT_EQ(labels(1, 0), "B");
//     EXPECT_EQ(labels(2, 0), "C");
// }

TEST_F(UtilsTest, LoadSingleRowCSV) {
    auto [features, labels] = loadCSVToMarray<int>(singleRowCSV.string(), ',');
    
    ASSERT_EQ(features.shape(0), 1);
    ASSERT_EQ(features.shape(1), 2);
    ASSERT_EQ(labels.shape(0), 1);
    
    EXPECT_EQ(features(0, 0), 1);
    EXPECT_EQ(features(0, 1), 2);
    EXPECT_EQ(labels(0, 0), 3);
}

// TEST_F(UtilsTest, LoadInvalidCSVThrowsException) {
   
// }

// TEST_F(UtilsTest, LoadNonExistentFileThrowsException) {
//     EXPECT_THROW(
//         loadCSVToMarray<int, int>("nonexistent.csv", ','),
//         std::runtime_error
//     )
// }

// TEST_F(UtilsTest, LoadHeaderWithEmptyRows) {
//     auto [features, labels] = loadCSVToMarray<int, int>(headerEmptyCSV.string(), ',');
    
//     // Should detect header and skip empty row, leaving 2 valid rows
//     ASSERT_EQ(features.shape(0), 2);
//     ASSERT_EQ(features.shape(1), 2);
    
//     EXPECT_EQ(features(0, 0), 1);
//     EXPECT_EQ(features(1, 0), 4);
// }

// TEST_F(UtilsTest, LoadCSVToMarrayFileWithHeader) {
//     auto [features, labels] = loadCSVToMarray<float,float>(headerCSV.string(), ',');
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
//     ASSERT_EQ(labels.shape(0), 3);
//     ASSERT_EQ(features(0, 0), 1.0);
//     ASSERT_EQ(features(0, 1), 2.0);
//     ASSERT_EQ(features(1, 0), 4.0);
//     ASSERT_EQ(features(1, 1), 5.0);
//     ASSERT_EQ(features(2, 0), 7.0);
//     ASSERT_EQ(features(2, 1), 8.0);
//     ASSERT_EQ(labels.shape(1), 1); // No labels in this case
//     ASSERT_EQ(labels(0, 0), 3.0); // Last column is considered as label
//     ASSERT_EQ(labels(1, 0), 6.0);
//     ASSERT_EQ(labels(2, 0), 9.0);

// }

// TEST_F(UtilsTest, LoadCSVToMarrayFileWithEmptyRow) {
//     auto [features, labels] = loadCSVToMarray<int, int>(emptyRowCSV.string(), ',');
//     ASSERT_EQ(features.shape(0), 3);
//     ASSERT_EQ(features.shape(1), 2);
//     ASSERT_EQ(labels.shape(0), 3);
//     ASSERT_EQ(features(0, 0), 1);
//     ASSERT_EQ(features(0, 1), 2);
//     ASSERT_EQ(features(1, 0), 4);
//     ASSERT_EQ(features(1, 1), 5);
//     ASSERT_EQ(features(2, 0), 7);
//     ASSERT_EQ(features(2, 1), 8);
//     ASSERT_EQ(labels.shape(1), 1); // No labels in this case
//     ASSERT_EQ(labels(0, 0), 3); // Last column is considered as label
//     ASSERT_EQ(labels(1, 0), 6);
//     ASSERT_EQ(labels(2, 0), 9);
// }

