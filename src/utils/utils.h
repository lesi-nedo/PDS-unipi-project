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

template<typename T>
concept Primitive = std::is_arithmetic_v<T>;

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
            
        Feature value;
        auto res = std::from_chars(sv.data(), sv.data() + sv.size(), value);
        if (res.ec != std::errc()) {
            hasHeader = true;
            std::cout << "Detected header in file: " << filename << std::endl;
        }

        ++max_columns;
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
    std::vector<size_t> featuresShape = {numRows, numFeatures};
    std::vector<size_t> labelsShape = {numRows, numLabels};

    // Initialize arrays with proper default values
    andres::Marray<Feature> features(featuresShape.begin(), featuresShape.end());
    andres::Marray<int> labels(labelsShape.begin(), labelsShape.end(), -1); // Default label as 0
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
            if (res.ec != std::errc()) {
                labels(i, 0) = labelValue;
            } else {
                if (auto it = labelMap.find(labelStr.data()); it == labelMap.end()) {
                    labelMap[labelStr.data()] = nex_int_label_id++;
                    labels(i, 0) = labelMap[labelStr.data()];
                } else {
                    labels(i, 0) = it->second;
                }
            }
            
            
        }
    }
    
    return {std::move(features), std::move(labels)};
}


#endif