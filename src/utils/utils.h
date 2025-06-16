#ifndef MY_RF_UTILS_H
#define MY_RF_UTILS_H
#include "../../include/sequential/marray.h"
#include <fstream>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <utility>
#include <ranges>
#include <variant>
#include <map>
#include <charconv>
#include <filesystem>
#include <cmath>
#include <iomanip>

template<typename T>
concept Primitive = std::is_arithmetic_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

/**
 * @brief Loads feature and label data from a CSV file into Marray containers.
 * 
 * This template function reads a CSV file and parses it into separate feature and label
 * arrays. It automatically detects headers, handles missing values, and converts string
 * labels to integer mappings. The function assumes the last column contains labels and
 * all preceding columns contain features.
 * 
 * @tparam Feature Primitive type for feature data (must satisfy Primitive concept)
 * 
 * @param filename The path and name of the input CSV file
 * @param delimeter The character used to separate values in the CSV file
 * 
 * @return std::pair<andres::Marray<Feature>, andres::Marray<int>> A pair containing:
 *         - First: Marray of features (2D array: rows x features)
 *         - Second: Marray of integer labels (2D array: rows x 1)
 * 
 * @throws std::runtime_error If the file cannot be opened
 * @throws std::runtime_error If the file is empty or cannot be read
 * @throws std::runtime_error If no valid data is found in the file
 * 
 * @note The function automatically detects headers by attempting to parse feature columns as numbers
 * @note Missing values (NaN, N/A, null, NULL) are skipped and the row is excluded
 * @note String labels are automatically mapped to integers starting from 0
 * @note Rows with inconsistent column counts are excluded from processing
 * @note The last column is always treated as the label column
 * 
 * @example
 * @code
 * auto [features, labels] = loadCSVToMarray<double>("dataset.csv", ',');
 * // features contains all columns except the last as double values
 * // labels contains the last column converted to integers
 * @endcode
 */
template<Primitive Feature>                    
std::pair<andres::Marray<Feature>, andres::Marray<int>> loadCSVToMarray(const std::string_view filename, const char delimeter) {

    std::ifstream file(filename.data());
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + std::string(filename));
    }
    
    std::string firstLine;
    bool hasHeader = false;
    int max_columns = 0;

    if (!std::getline(file, firstLine)) {
        throw std::runtime_error("File is empty or cannot be read: " + std::string(filename));
    }
        
    for(auto token: std::views::split(firstLine, delimeter)){
        std::string_view sv(&*token.begin(), std::distance(token.begin(), token.end()));
        if (sv.empty()) {
            continue;
        }
        ++max_columns;
    }

    // Check only the feature columns (exclude last column which is label)
    int featureColumnsParsed = 0;
    int currentColumn = 0;
    for(auto token: std::views::split(firstLine, delimeter)){
        std::string_view sv(&*token.begin(), std::distance(token.begin(), token.end()));
        if (sv.empty()) {
            continue;
        }
        
        // Only check feature columns (not the label column)
        if (currentColumn < max_columns - 1) {
            Feature value;
            auto res = std::from_chars(sv.data(), sv.data() + sv.size(), value);
            if (res.ec == std::errc()) {
                featureColumnsParsed++;
            }
        }
        currentColumn++;
    }

    // If most feature columns can't be parsed as numbers, likely a header
    if (featureColumnsParsed < (max_columns - 1) / 2) {
        hasHeader = true;
        std::cout << "Detected header in file: " << filename << std::endl;
    }

    file.clear();
    file.seekg(0, std::ios::beg);

    std::vector<std::vector<std::string>> rows;
    if (hasHeader) {
        std::getline(file, firstLine); // Skip header line
    }
    while (std::getline(file, firstLine)) {
        std::string_view lineView(firstLine);
        auto tokens = std::views::split(lineView, delimeter);
        
        int columnCount = 0;
        std::vector<std::string> row;
        for (auto token : tokens) {
            auto sv = std::string_view(&*token.begin(), std::distance(token.begin(), token.end()));
            if (sv.empty() || sv == "NaN" || sv == "nan" || sv == "N/A" || sv == "n/a" || sv == "null" || sv == "NULL") {
                break;
            }
            row.push_back(std::string(sv));
            columnCount++;
        }
        max_columns = std::max(max_columns, columnCount);
        if (columnCount != max_columns) {
            continue;
        }
        if (!row.empty()) {
            rows.push_back(std::move(row));
        }
    }

    file.close();

    if (rows.empty()) {
        throw std::runtime_error("No valid data found in file: " + std::string(filename));
    }

    const size_t numRows = rows.size();
    const size_t numFeatures = max_columns - 1; // Last column is label
    const size_t numLabels = 1; // Assuming one label column
    const size_t shape[] = {numRows, numFeatures};

    // Initialize arrays with proper default values
    andres::Marray<Feature> features(shape, shape+2);
    andres::Marray<int> labels(shape, shape+1); // Default label as 0
    std::map<std::string, int> labelMap;
    int nex_int_label_id = 0;
    for(size_t i = 0; i < numRows; ++i) {
        if (rows[i].size() == max_columns) {
            // Fill features
            for (size_t j = 0; j < numFeatures; ++j) {
                Feature value;
                if constexpr (std::is_floating_point_v<Feature>) {
                    value = std::stod(rows[i][j]);
                } else if constexpr (std::is_integral_v<Feature>) {
                    value = std::stoll(rows[i][j]);
                }
                features(i, j) = value;
            }
            
            const std::string_view labelStr = rows[i][numFeatures];
            int labelValue;
            auto res = std::from_chars(labelStr.data(), labelStr.data() + labelStr.size(), labelValue);
            if (res.ec == std::errc()) {
                labels(i) = labelValue;
            } else {
                if (auto it = labelMap.find(labelStr.data()); it == labelMap.end()) {
                    labelMap[labelStr.data()] = nex_int_label_id++;
                    labels(i) = labelMap[labelStr.data()];
                } else {
                    labels(i) = it->second;
                }
            }
            
            
        }
    }
    
    return {std::move(features), std::move(labels)};
}

/**
 * @brief Saves feature and label data from Marray containers to a CSV file.
 * 
 * This template function writes feature data and corresponding labels to a CSV file
 * with a specified delimiter. The features and labels are written row by row, with
 * features first followed by the label for each row.
 * 
 * @tparam Feature Primitive type for feature data (must satisfy Primitive concept)
 * @tparam Label Integral type for label data (must satisfy Integral concept)
 * 
 * @param features The Marray containing feature data (2D array: rows x features)
 * @param labels The Marray containing label data (2D array: rows x 1)
 * @param filename The path and name of the output CSV file
 * @param delimeter The character used to separate values (default: ',')
 * 
 * @throws std::runtime_error If the parent directory doesn't exist
 * @throws std::runtime_error If the file cannot be opened for writing
 * @throws std::runtime_error If features and labels have different number of rows
 * @throws std::runtime_error If features or labels have zero columns
 * @throws std::runtime_error If there are no rows to write
 * @throws std::runtime_error If labels have more than one column (only single label supported)
 * 
 * @note The function expects labels to be a single column (labels.shape(1) == 1)
 * @note The parent directory must exist before calling this function
 * 
 * @example
 * @code
 * andres::Marray<double> features({100, 5}); // 100 samples, 5 features
 * andres::Marray<int> labels({100, 1});      // 100 labels
 * saveMarrayToCSV(features, labels, "output.csv", ',');
 * @endcode
 */
template<Primitive Feature>
void saveMarrayToCSV(const andres::Marray<Feature>& features, 
                     const andres::Marray<int>& labels, 
                     const std::string_view filename,
                     const std::map<int, std::string>& labelMap = {},
                     const char delimeter = ','
) {

    std::filesystem::path path(filename);
    if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())) {
        throw std::runtime_error("Directory does not exist: " + path.parent_path().string() + 
                                ". Please create the directory first or use an existing path.");
    }
    
    std::ofstream file(filename.data());
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + std::string(filename));
    }
    if (features.shape(0) != labels.shape(0)) {
        throw std::runtime_error("Features and labels must have the same number of rows.");
    }
    const size_t numRows = features.shape(0);
    const size_t numFeatures = features.shape(1);
    const size_t dimLables = labels.dimension();
    if (numFeatures == 0 ) {
        throw std::runtime_error("Features must have at least one column.");
    }
    if (numRows == 0) {
        throw std::runtime_error("No data to write to file.");
    }
    if(dimLables > 1) {
        throw std::runtime_error("Labels must have exactly one column (labels.shape(1) == 1).");
    }

    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numFeatures; ++j) {
            Feature value = features(i, j);
            if constexpr (std::is_floating_point_v<Feature>)
                if (value == std::floor(value)) 
                    file << std::fixed << std::setprecision(1) << value;
                else 
                    file << std::fixed << std::setprecision(6) << value;
            else
                file << value;
            
            if (j < numFeatures - 1) {
                file << delimeter;
            }
        }
        auto it = labelMap.find(labels(i));
        if (it != labelMap.end()) {
            file << delimeter << it->second; // Write mapped label
        } else {
            file << delimeter << static_cast<long long>(labels(i)); // Fallback to direct label
        }
        file << '\n';
    }
    file.close();

}                 

/**
 * @brief Count the number of unique labels in a label array
 * @param labels The Marray containing label data
 * @return The number of unique labels
 */
size_t countUniqueLabels(const andres::Marray<int>& labels);

#endif