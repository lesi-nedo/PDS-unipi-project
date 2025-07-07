#include "marray.h" 
#include "utils.h"
#include "dt_ff_exp2.h"
#include "general_config.h"


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <train_dataset_path> <test_dataset_path>" << std::endl;
        return 1;
    }

    using namespace andres;

    std::string train_dataset = argv[1];
    std::string test_dataset = argv[2];

    auto curr_path = std::filesystem::current_path();
    
    if (curr_path.filename() != "project") {
        std::cerr << "Error: Please run this program from the 'project' directory." << std::endl;
        return 1;
    }
    std::string results_path = "./results/validation/ff_sklearn/";
    if(std::filesystem::exists(results_path)) {
        std::filesystem::remove_all(results_path);
    }
    std::filesystem::create_directory(results_path);
    std::cout << "Results will be saved to: " << results_path << std::endl;
    ff_ml_exp2::DecisionForest<double, int, double> rf;
    const int randomSeed = 42; // Fixed seed for reproducibility
    const std::vector<size_t> samples_perTree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> samples_perTree_test = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;

    for(auto& samples_perTree_i : samples_perTree) {
        
        auto [features_train, labels_train] = loadCSVToMarray<double>(train_dataset, ',', samples_perTree_i, andres::FirstMajorOrder);
        auto [features_test, labels_test] = loadCSVToMarray<double>(test_dataset, ',', samples_perTree_test[0], andres::LastMajorOrder);

        for(auto& tree_counts_i : tree_counts) {
            rf.clear();

            std::cout << "Training with " << tree_counts_i << " trees and " << samples_perTree_i << " samples per tree." << std::endl;
            rf.learnWithFFNetwork(features_train, labels_train, tree_counts_i, randomSeed);

            const size_t shape[] = {features_test.shape(0), countUniqueLabels(labels_test)};
            andres::Marray<double> predictions(shape, shape+2);

            std::cout << "Predicting with " << tree_counts_i << " trees." << std::endl;
            rf.predict(features_test, predictions);
            
            std::cout << "Predictions shape: " << predictions.shape(0) << " x " << predictions.shape(1) << std::endl;
            // Convert probabilities to class predictions
            std::vector<int> classPredictions = probabilitiesToPredictions(predictions);

            double accuracy = calculateAccuracy(classPredictions, labels_test);
            std::cout << "Training accuracy with " << tree_counts_i << " trees: " 
                      << std::fixed << std::setprecision(4) << accuracy << std::endl;
            try {
                savePredictionsToFile(
                    classPredictions,
                    results_path + "predictions_" + std::to_string(tree_counts_i) + "_trees_" + 
                    std::to_string(samples_perTree_i) + "_samples.csv"
                );
                saveProbabilitiesToFile(
                    predictions,
                    results_path + "probabilities_" + std::to_string(tree_counts_i) + "_trees_" + 
                    std::to_string(samples_perTree_i) + "_samples.csv"
                );
                std::cout << "Saved predictions and probabilities to files." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error saving files: " << e.what() << std::endl;
                return 1;
            }

            std::cout << "----------------------------------------" << std::endl;
            std::cout << "\nFirst 10 predictions:" << std::endl;
            std::cout << "Samples | Actual | Predicted | Probabilities" << std::endl;
            std::cout << "--------|--------|-----------|---------------" << std::endl;
            for (size_t i = 0; i < std::min(static_cast<size_t>(10), predictions.shape(0)); ++i) {
                std::cout << "   " << i + 1 << "    | "
                          << labels_test(i) << "    | "
                          << "   " << classPredictions[i] << "    | ";
                for (size_t j = 0; j < predictions.shape(1); ++j) {
                    std::cout << std::fixed << std::setprecision(3) << predictions(i, j);
                    if (j < predictions.shape(1) - 1) std::cout << " ";
                }
                std::cout << std::endl;
            }


        }
    }
    std::cout << "All tests completed successfully." << std::endl;
    return 0;
}