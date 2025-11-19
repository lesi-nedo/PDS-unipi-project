#ifndef MY_RF_UTILS_H
#define MY_RF_UTILS_H

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
#include <unordered_set>

#include "marray.h"
#include "general_config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

template<typename T>
concept Primitive = std::is_arithmetic_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

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

template<typename Forest, typename Feature, typename Label, typename Probability>
concept DecisionForestConceptMPI = requires (
    Forest forest,
    const size_t num_train_samples,
    const size_t num_test_samples,
    const size_t numberOfTrees,
    const int world_size,
    const int world_rank,
    const int randomSeed,
    const std::string_view train_dataset_path,
    const std::string_view test_dataset_path,
    andres::Marray<Probability>& probabilities
){
    { forest.learnMaster(num_train_samples, numberOfTrees, world_size, randomSeed) };
    { forest.predictMaster(num_test_samples, world_size, probabilities) };
    { forest.size() } -> std::convertible_to<size_t>;
    { forest.terminateTraining(world_size) };
    { forest.terminatePrediction(world_size) };
    { forest.learnWorker(train_dataset_path, world_rank, randomSeed) };
    { forest.predictWorker(test_dataset_path, world_rank) };
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
std::pair<andres::Marray<Feature>, andres::Marray<int>> loadCSVToMarray(
    const std::string_view filename, const char delimeter=',', int max_rows=-1,
    const andres::CoordinateOrder order = andres::LastMajorOrder
) {

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
    andres::Marray<Feature> features(shape, shape+2, Feature(), order);
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
template<typename Label>
size_t countUniqueLabels(const andres::Marray<Label>& labels) {
    std::unordered_set<Label> uniqueLabels;
    for (size_t i = 0; i < labels.shape(0); ++i) {
        uniqueLabels.insert(labels(i));
    }
    return uniqueLabels.size();
}

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


inline void saveTrainingDataHeader(
    std::ofstream &performance_log
){
    performance_log << "NumTrees,TrainingTime_ms,MemoryUsage_MB,TrainSamples,TrainingThroughput_samples_per_sec,\n";

}

inline void addTrainRowToLog(
    std::ofstream &performance_log,
    const std::tuple<size_t, double, double, size_t, double> &training_data
){
    
    performance_log << std::get<0>(training_data) << ","
                    << std::get<1>(training_data) << ","
                    << std::get<2>(training_data) << ","
                    << std::get<3>(training_data) << ","
                    << std::get<4>(training_data) << ",";
}

inline void savePredictionDataHeader(
    std::ofstream &performance_log
){
    performance_log << "PredictionTime_ms,TestSamples,FeaturesPerTree,PredictionThroughput_samples_per_sec,Accuracy,F1Score,Precision,Recall\n";

}

inline void addPredictionRowToLog(
    std::ofstream &performance_log,
    std::tuple<double, size_t, size_t, double, double, double, double, double> prediction_data
){
    
    performance_log << std::get<0>(prediction_data) << ","
                    << std::get<1>(prediction_data) << ","
                    << std::get<2>(prediction_data) << ","
                    << std::get<3>(prediction_data) << ","
                    << std::get<4>(prediction_data) << ","
                    << std::get<5>(prediction_data) << ","
                    << std::get<6>(prediction_data) << ","
                    << std::get<7>(prediction_data) << "\n";
}


/**
 * @brief Helper function to set thread count based on implementation
 */
inline void setThreadCount(int num_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
    // For FastFlow, thread count is typically set via configuration
    std::cout << "Setting thread count to: " << num_threads << std::endl;
}

/**
 * @brief Calculate comprehensive metrics including accuracy, F1, precision, and recall
 */
template<typename Probability, typename Label>
std::tuple<double, double, double, double> calculateMetrics(
    const andres::Marray<Probability>& probabilities,
    const andres::Marray<Label>& labels_test
) {
    auto predicted_classes = probabilitiesToPredictions(probabilities);
    double accuracy = calculateAccuracy(predicted_classes, labels_test);
    auto [precision, recall, f1] = calculatePrecisionRecallF1(predicted_classes, labels_test);
    return {accuracy, f1, precision, recall};
}

/**
 * @brief Extended performance evaluation with systematic parameter variation
 * 
 * This function performs comprehensive parameter sweeps
 * for thorough performance analysis including thread count variations.
 */
template<
    typename Feature,
    typename Label,
    typename Probability,
    typename ForestType,
    typename TrainFn,
    typename... Ts
>
void run_comprehensive_evaluation(
    const std::vector<size_t>& tree_counts,
    const std::vector<size_t>& samples_per_tree,
    const std::vector<size_t>& samples_per_tree_test,
    const std::vector<std::tuple<Ts...>>& thread_counts,
    const int num_pred_workers,
    const std::string_view train_dt_path,
    const std::string_view test_dt_path,
    const std::string& results_path,
    ForestType& forest,
    TrainFn train_fn,
    const size_t num_of_features,
    const int random_seed = 0
) {

    if (thread_counts.empty()) {
        throw std::runtime_error("Thread counts vector cannot be empty.");
    }
    // const auto first_tuple = std::get<0>(thread_counts);
    
    using TupleType = typename std::vector<std::tuple<Ts...>>::value_type;
    constexpr size_t num_elements = std::tuple_size<TupleType>::value;
    
    static_assert(num_elements == 3 || num_elements == 5,
                  "Tuple must have 3 or 5 elements");

    std::filesystem::create_directories(results_path);
    std::ofstream performance_log(results_path + "comprehensive_performance.csv");
    
    // Enhanced CSV header with thread count and phase-specific metrics
    performance_log << "NumTrees,NumThreads,TrainingTime_ms,PredictionTime_ms,"
                    << "TotalTime_ms,Accuracy,MemoryUsage_MB,SamplesPerTree,"
                    << "FeaturesPerTree,TrainingThroughput_samples_per_sec,"
                    << "PredictionThroughput_samples_per_sec,F1Score,Precision,Recall,"
                    << "TrainingSpeedup,PredictionSpeedup,OverallSpeedup,"
                    << "TrainingEfficiency,PredictionEfficiency,OverallEfficiency\n";
    
    // Store baseline (single-thread) times for speedup calculation
    std::unordered_map<std::string, std::tuple<double, double>> baseline_times;
    
    for(const auto& num_threads : thread_counts) {
        
        // Convert tuple to vector for runtime access
        auto all_threads_vec = std::apply([](auto&&... args) {
            return std::vector<int>{static_cast<int>(args)...};
        }, num_threads);

        std::vector<int> threads_pred = {};
        auto total_pred_threads = 0;
        
        // Extract prediction threads (last num_pred_workers)
        for(size_t i = 1; i <= static_cast<size_t>(num_pred_workers); ++i) {
            if (num_elements >= i) {
                int val = all_threads_vec[num_elements - i];
                threads_pred.push_back(val);
                total_pred_threads += val;
            }
        }
        
        // Extract rest threads (first num_elements - num_pred_workers)
        std::vector<int> rest_thread_counts_vec;
        if (num_elements > static_cast<size_t>(num_pred_workers)) {
            for(size_t k = 0; k < num_elements - static_cast<size_t>(num_pred_workers); ++k) {
                rest_thread_counts_vec.push_back(all_threads_vec[k]);
            }
        }
        
        int total_threads = 1;
        if (num_elements >= 2) {
            int sum = 0;
            for(int t : rest_thread_counts_vec) sum += t;
            total_threads = sum > 0 ? sum : 1;
        }
        for(size_t idx = 0; idx < samples_per_tree.size(); ++idx) {
            auto samples_train = samples_per_tree[idx];
            auto samples_test = samples_per_tree_test[idx];
            
            auto [features_train, labels_train] = loadCSVToMarray<Feature>(
                train_dt_path, ',', samples_train, andres::LastMajorOrder
            );
            auto [features_test, labels_test] = loadCSVToMarray<Feature>(
                test_dt_path, ',', samples_test, andres::LastMajorOrder
            );
            
            const size_t num_features = features_train.shape(1);
            
            for(const auto& numberOfTrees : tree_counts) {
                std::string config_key = std::to_string(numberOfTrees) + "_" + std::to_string(samples_train);
                
                std::cout << "\n=== Configuration: Trees=" << numberOfTrees 
                          << ", Threads=" << total_threads
                          << ", TrainSamples=" << samples_train << " ===" << std::endl;
                
                long memory_before = getMemoryUsageMB();
                
                // TRAINING PHASE - Separate timing
                auto start_train = std::chrono::high_resolution_clock::now();
                train_fn(forest, features_train, labels_train, numberOfTrees, rest_thread_counts_vec);
                auto end_train = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> train_duration = end_train - start_train;
                
                long memory_after = getMemoryUsageMB();
                long memory_usage = memory_after - memory_before;
                
                // PREDICTION PHASE - Separate timing
                const size_t shape[] = {features_test.shape(0), num_of_features};
                andres::Marray<Probability> probabilities(shape, shape + 2);
                
                auto start_predict = std::chrono::high_resolution_clock::now();
                forest.predict(features_test, probabilities, threads_pred);
                auto end_predict = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> predict_duration = end_predict - start_predict;
                
                // Calculate metrics
                double total_time = train_duration.count() + predict_duration.count();
                double training_throughput = (samples_train * 1000.0) / train_duration.count();
                double prediction_throughput = (samples_test * 1000.0) / predict_duration.count();
                
                auto [accuracy, f1, precision, recall] = calculateMetrics(
                    probabilities, labels_test
                );
                
                // Calculate speedup and efficiency
                double train_speedup = 1.0, pred_speedup = 1.0, overall_speedup = 1.0;
                double train_efficiency = 1.0, pred_efficiency = 1.0, overall_efficiency = 1.0;
                bool is_baseline = true;
                
                is_baseline = std::all_of(rest_thread_counts_vec.begin(), rest_thread_counts_vec.end(), [](int val){ return val == 1; });
                
                if (is_baseline) {
                    baseline_times[config_key] = {train_duration.count(), predict_duration.count()};
                } else if (baseline_times.count(config_key)) {
                    auto [baseline_train, baseline_pred] = baseline_times[config_key];
                    train_speedup = baseline_train / train_duration.count();
                    pred_speedup = baseline_pred / predict_duration.count();
                    overall_speedup = (baseline_train + baseline_pred) / total_time;
                    
                    train_efficiency = train_speedup / total_threads;
                    pred_efficiency = pred_speedup / total_pred_threads;
                    overall_efficiency = overall_speedup / (total_threads + total_pred_threads);
                }

                forest.clear();
                
                // Log comprehensive results
                performance_log << numberOfTrees << ","
                              << total_threads << ","
                              << train_duration.count() << ","
                              << predict_duration.count() << ","
                              << total_time << ","
                              << accuracy << ","
                              << memory_usage << ","
                              << samples_train << ","
                              << num_features << ","
                              << training_throughput << ","
                              << prediction_throughput << ","
                              << f1 << ","
                              << precision << ","
                              << recall << ","
                              << train_speedup << ","
                              << pred_speedup << ","
                              << overall_speedup << ","
                              << train_efficiency << ","
                              << pred_efficiency << ","
                              << overall_efficiency << "\n";
                
                performance_log.flush();
                
                std::cout << "Training Time: " << train_duration.count() << " ms" << std::endl;
                std::cout << "Prediction Time: " << predict_duration.count() << " ms" << std::endl;
                std::cout << "Training Speedup: " << train_speedup << "x" << std::endl;
                std::cout << "Prediction Speedup: " << pred_speedup << "x" << std::endl;
                std::cout << "Training Efficiency: " << (train_efficiency * 100) << "%" << std::endl;
                std::cout << "Prediction Efficiency: " << (pred_efficiency * 100) << "%" << std::endl;
                std::cout << "Accuracy: " << accuracy << std::endl;
            }
        }
    }
    
    performance_log.close();
    std::cout << "\nComprehensive evaluation finished. Results saved to " 
              << results_path + "comprehensive_performance.csv" << std::endl;
}





#endif