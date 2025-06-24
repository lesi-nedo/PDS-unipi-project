#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "dt.h"
#include "marray.h"
#include "utils.h"




int main(int argc, char* argv[]) {
    using namespace andres;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <train_dataset_path> <test_dataset_path>" << std::endl;
        return 1;
    }

    std::string train_dataset_path = argv[1];
    std::string test_dataset_path = argv[2];
    auto curr_path = std::filesystem::current_path();
    std::cout << "CURRENT WORKING DIRECTORY: " << curr_path << std::endl;
    
    if (curr_path.filename() != "project") {
        std::cerr << "Error: Please run this program from the 'project' directory." << std::endl;
        return 1;
    }
    const std::string results_path = curr_path / "results" / "sequential" / "";

    // Ensure the results directory exists
    std::filesystem::create_directories(results_path);

    const std::vector<size_t> samples_perTree = {50, 100, 150, 200, 250};
    // Setup for performance logging
    std::ofstream performance_log(results_path + "performance.csv");
    performance_log << "NumTrees,TrainingTime_ms,PredictionTime_ms,Accuracy,MemoryUsage_MB,SamplesPerTree,FeaturesPerTree,TrainingThroughput_samples_per_sec,PredictionThroughput_samples_per_sec,F1Score,Precision,Recall\n";
    std::cout << "\nStarting performance evaluation loop..." << std::endl;

    for(const auto& samples : samples_perTree) {
        auto [features_train, labels_train] = loadCSVToMarray<double>(train_dataset_path, ',', samples); 
        auto [features_test, labels_test] = loadCSVToMarray<double>(test_dataset_path, ',', samples);

        std::cout << "Train Features shape: " << features_train.shape(0) << " x " << features_train.shape(1) << std::endl;
        std::cout << "Train Labels shape: " << labels_train.shape(0) << std::endl;


        const std::vector<size_t> tree_counts = {10, 50, 100, 150, 200};


        for (const auto& numberOfTrees : tree_counts) {
            std::cout << "\n--- Testing with " << numberOfTrees << " trees ---" << std::endl;

            // Create a DecisionForest instance
            ml::DecisionForest<double, int, double> forest;
            std::random_device rd;
            // Reset random seed for each run for consistent results
            std::mt19937 randomEngine(rd());
            long memory_before = getMemoryUsageMB();

            // Time the training phase
            std::cout << "Learning decision trees..." << std::endl;
            auto start_train = std::chrono::high_resolution_clock::now();
            forest.learn(features_train, labels_train, numberOfTrees, randomEngine);
            auto end_train = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> train_duration = end_train - start_train;
            std::cout << "Learned " << forest.size() << " decision trees in " << train_duration.count() << " ms." << std::endl;
            long memory_after = getMemoryUsageMB();
            double memrory_usage = static_cast<double>(memory_after - memory_before);

            double trainingThroughput = features_train.shape(0) / (train_duration.count() / 1000.0);
            
            // Prepare for prediction
            const size_t shape[] = {features_test.shape(0), countUniqueLabels(labels_test)};
            Marray<double> predictions(shape, shape+2);

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
                            << memrory_usage << ","
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

    return 0;
}