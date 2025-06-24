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
#include <sys/resource.h>
#include <unordered_map>
#include <algorithm>
#include <random>


template<typename T>
concept Primitive = std::is_arithmetic_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;


template<typename Forest, typename Feature, typename Label, typename Probability>
concept DecisionForestConceptEngine = requires(
    Forest forest,
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    size_t numberOfTrees,
    std::mt19937& randomEngine,
    andres::Marray<Probability>& probabilities
) {
    { forest.learn(features, labels, numberOfTrees, randomEngine) };
    { forest.predict(features, probabilities) };
    { forest.size() } -> std::convertible_to<size_t>;
};

template<typename Forest, typename Feature, typename Label, typename Probability>
concept DecisionForestConceptSeed = requires(
    Forest forest,
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    size_t numberOfTrees,
    const int randomSeed,
    andres::Marray<Probability>& probabilities
) {
    { forest.learn(features, labels, numberOfTrees, randomSeed) };
    { forest.predict(features, probabilities) };
    { forest.size() } -> std::convertible_to<size_t>;
};

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
std::pair<andres::Marray<Feature>, andres::Marray<int>> loadCSVToMarray(const std::string_view filename, const char delimeter=',', int max_rows=-1) {

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

    if (max_rows == 0) {
        max_rows = -1; // No limit
    }
    int rowCount = 0;
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
        rowCount++;
        if (max_rows > 0 && rowCount >= max_rows) {
            break; // Stop reading if max_rows limit is reached
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

/**
 * @brief Convert probability predictions to class predictions
 *  @param probabilities The Marray containing probability predictions (2D array: samples x classes)
 * @return A vector of class predictions
 *  @throws std::runtime_error If the probabilities array is empty
 */
std::vector<int> probabilitiesToPredictions(const andres::Marray<double>& probabilities);

/**
 * @brief Calculate the accuracy of predictions against actual labels
 * @param predicted A vector of predicted class labels
 * @param actual A View containing actual class labels
 * @return The accuracy as a double value between 0.0 and 1.0
 * @throws std::runtime_error If the sizes of predicted and actual labels do not match
 */
double calculateAccuracy(const std::vector<int>& predicted, const andres::View<int>& actual);

/**
 * @brief Save predictions to a text file
 * @param predictions A vector of predicted class labels
 * @param filename The path and name of the output file
 * @throws std::runtime_error If the file cannot be opened for writing
 */
void savePredictionsToFile(const std::vector<int>& predictions, const std::string& filename);

/**
 * @brief Save probability predictions to a CSV file
 * @param probabilities The Marray containing probability predictions (2D array: samples x classes)
 * @param filename The path and name of the output CSV file
 * @throws std::runtime_error If the file cannot be opened for writing
 */
void saveProbabilitiesToFile(const andres::Marray<double>& probabilities, const std::string& filename);

/**
 * @brief Count unique labels in a label array
 * @param labels The Marray containing label data
 * @return The number of unique labels
 * @throws std::runtime_error If the labels array is empty
 */
size_t countUniqueLabels(const andres::Marray<int>& labels);

/**
* @brief Get the current memory usage of the process in MB
* @return Memory usage in MB
*/

long getMemoryUsageMB();

/**
 * @brief Calculate precision, recall, and F1 score for predictions against true labels
 * @param predictions A vector of predicted class labels
 * @param true_labels A vector of true class labels
 * @return A tuple containing precision, recall, and F1 score as doubles
 * @throws std::invalid_argument If predictions and true_labels have different sizes
 * 
 * 
 */
std::tuple<double, double, double> calculatePrecisionRecallF1(
    const std::vector<int>& predictions, 
    const andres::Marray<int>& true_labels);



template<typename TrainFn, typename ForestType>
void run_test_impl(
    const std::vector<size_t>& tree_counts,
    const std::vector<size_t>& samples_per_tree,
    const std::string_view train_dt_path,
    const std::string_view test_dt_path,
    const std::string& results_path,
    ForestType& forest,
    TrainFn train_fn
) {
    std::filesystem::create_directories(results_path);
    std::ofstream performance_log(results_path + "performance.csv");
    performance_log << "NumTrees,TrainingTime_ms,PredictionTime_ms,Accuracy,MemoryUsage_MB,SamplesPerTree,FeaturesPerTree,TrainingThroughput_samples_per_sec,PredictionThroughput_samples_per_sec,F1Score,Precision,Recall\n";
    std::cout << "\nStarting performance evaluation loop..." << std::endl;

    for(const auto& samples : samples_per_tree) {
        auto [features_train, labels_train] = loadCSVToMarray<double>(train_dt_path, ',', samples); 
        auto [features_test, labels_test] = loadCSVToMarray<double>(test_dt_path, ',', samples);
        
        std::cout << "--- Loaded datasets ---" << std::endl ;
        std::cout << "Train Features shape: " << features_train.shape(0) << " x " << features_train.shape(1) << std::endl;
        std::cout << "Train Labels shape: " << labels_train.shape(0) << std::endl;

        for (const auto& numberOfTrees : tree_counts) {
            std::cout << "\n--- Testing with " << numberOfTrees << " trees ---" << std::endl;

            long memory_before = getMemoryUsageMB();

            // Time the training phase
            std::cout << "Learning decision trees..." << std::endl;
            auto start_train = std::chrono::high_resolution_clock::now();
            train_fn(forest, features_train, labels_train, numberOfTrees);
            auto end_train = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> train_duration = end_train - start_train;
            std::cout << "Learned " << forest.size() << " decision trees in " << train_duration.count() << " ms." << std::endl;
            long memory_after = getMemoryUsageMB();
            double memory_usage = static_cast<double>(memory_after - memory_before);

            double trainingThroughput = features_train.shape(0) / (train_duration.count() / 1000.0);

            // Prepare for prediction
            const size_t shape[] = {features_test.shape(0), countUniqueLabels(labels_test)};
            andres::Marray<double> predictions(shape, shape+2);

            // Time the prediction phase
            std::cout << "Predicting..." << std::endl;
            auto start_predict = std::chrono::high_resolution_clock::now();
            forest.predict(features_test, predictions);
            auto end_predict = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> predict_duration = end_predict - start_predict;
            double predictionThroughput = features_test.shape(0) / (predict_duration.count() / 1000.0);
            std::cout << "Prediction finished in " << predict_duration.count() << " ms." << std::endl;

            // Calculate and display accuracy
            std::vector<int> classPredictions = probabilitiesToPredictions(predictions);
            double accuracy = calculateAccuracy(classPredictions, labels_test);
            auto [precision, recall, f1] = calculatePrecisionRecallF1(classPredictions, labels_test);
            std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl << std::endl;

            // Log results to the CSV file
            performance_log << numberOfTrees << ","
                            << train_duration.count() << ","
                            << predict_duration.count() << ","
                            << accuracy << ","
                            << memory_usage << ","
                            << features_train.shape(0) << ","
                            << features_train.shape(1) << ","
                            << trainingThroughput << ","
                            << predictionThroughput << ","
                            << f1 << ","
                            << precision << ","
                            << recall << "\n";
        }
    }
    performance_log.close();
    std::cout << "\nPerformance evaluation finished. Results saved to " << results_path + "performance.csv" << std::endl;
}

/**
 * @brief Run a test for a decision forest engine with specified parameters
 * 
 * This function runs a series of tests on a decision forest engine using the provided
 * training and testing datasets. It measures training time, prediction time, accuracy,
 * and memory usage, and logs the results to a CSV file.
 * 
 * @tparam ForestType The type of the decision forest engine (must satisfy DecisionForestConceptEngine)
 * 
 * @param tree_counts A vector of tree counts to test
 * @param samples_per_tree A vector of sample sizes for each tree
 * @param train_dt_path Path to the training dataset in CSV format
 * @param test_dt_path Path to the testing dataset in CSV format
 * @param results_path Path to save the performance results in CSV format
 * @param forest The decision forest engine instance to be tested
 * @param randomEngine A random number generator engine to be used for training
 * 
 * @throws std::runtime_error If the training or testing datasets cannot be loaded
 * @throws std::runtime_error If the results file cannot be opened for writing
 * 
 * @note The function assumes the last column of the training and testing datasets contains labels
 * @note The function will log the results in a CSV format with columns for tree count,
 *       samples per tree, training time, prediction time, accuracy, and memory usage
 * @note The function will also print the results to the console for immediate feedback
 * * @example
 * @code
 * DecisionForestEngine forest;
 * std::vector<size_t> tree_counts = {10, 20, 30};
 * std::vector<size_t> samples_per_tree = {100, 200, 300};
 * run_test(tree_counts, samples_per_tree, "train_data.csv", "test_data.csv", "results.csv", forest);
 * @endcode
 */
template<
    typename Feature,
    typename Label,
    typename Probability,
    typename RandomEngine,
    DecisionForestConceptEngine<Feature, Label, Probability> ForestType
>
void run_test(
    const std::vector<size_t>& tree_counts,
    const std::vector<size_t>& samples_per_tree,
    const std::string_view train_dt_path,
    const std::string_view test_dt_path,
    const std::string& results_path,
    ForestType& forest,
    RandomEngine& randomEngine
) {
    
    run_test_impl(
        tree_counts, samples_per_tree, train_dt_path, test_dt_path, results_path, forest,
        [&](auto& forest, const auto& features, const auto& labels, size_t numberOfTrees) {
            forest.learn(features, labels, numberOfTrees, randomEngine);
        }
    );
}

/**
 * @brief Run a test for a decision forest engine with specified parameters
 * This function runs a series of tests on a decision forest engine using the provided
 * training and testing datasets. It measures training time, prediction time, accuracy,
 * and memory usage, and logs the results to a CSV file.
 * @param tree_counts A vector of tree counts to test
 * @param samples_per_tree A vector of sample sizes for each tree
 * @param train_dt_path Path to the training dataset in CSV format
 * @param test_dt_path Path to the testing dataset in CSV format
 * @param results_path Path to save the performance results in CSV format
 * @param forest The decision forest engine instance to be tested
 * @param random_seed A random seed to be used for training (default: 0)
 * 
 * @throws std::runtime_error If the training or testing datasets cannot be loaded
 * @throws std::runtime_error If the results file cannot be opened for writing
 * @note The function assumes the last column of the training and testing datasets contains labels
 * @note The function will log the results in a CSV format with columns for tree count,
 *      samples per tree, training time, prediction time, accuracy, and memory usage
 * @note The function will also print the results to the console for immediate feedback
 * * @example
 * @code
 * DecisionForestEngine forest;
 * std::vector<size_t> tree_counts = {10, 20, 30};
 * std::vector<size_t> samples_per_tree = {100, 200, 300};
 * run_test(tree_counts, samples_per_tree, "train_data.csv", "test_data.csv", "results.csv", forest, 42);
 * @endcode
 */
template<
    typename Feature,
    typename Label,
    typename Probability,
    DecisionForestConceptSeed<Feature, Label, Probability> ForestType
>
void run_test(
    const std::vector<size_t>& tree_counts,
    const std::vector<size_t>& samples_per_tree,
    const std::string_view train_dt_path,
    const std::string_view test_dt_path,
    const std::string& results_path,
    ForestType& forest,
    const int random_seed=0
) {
    run_test_impl(
        tree_counts, samples_per_tree, train_dt_path, test_dt_path, results_path, forest,
        [&](auto& forest, const auto& features, const auto& labels, size_t numberOfTrees) {
            forest.learn(features, labels, numberOfTrees, random_seed);
        }
    );
}

#endif