#include <stdexcept>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "../../include/sequential/marray.h"
#include "../../include/sequential/dt.h"
#include "utils.h"




int main(int argc, char* argv[]) {
    using namespace andres;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <train_dataset_path> <test_dataset_path>" << std::endl;
        return 1;
    }

    std::string train_dataset = argv[1];
    std::string test_dataset = argv[2];
    std::cout <<  "CURRENT WORKING DIRECTORY: " << std::filesystem::current_path() << std::endl;
    const std::string path {"./results/validation/seq_sklearn/"};

    auto [features_train, labels_train] = loadCSVToMarray<double>(train_dataset, ',');
    auto [features_test, labels_test] = loadCSVToMarray<double>(test_dataset, ',');


    std::cout << "Train Features shape: " << features_train.shape(0) << " x " << features_train.shape(1) << std::endl;
    std::cout << "Train Labels shape: " << labels_train.shape(0) << std::endl;

    // Create a DecisionForest instance with fixed random seed for reproducibility
    ml::DecisionForest<double, int, double> forest;
    const size_t numberOfTrees = 100; // Use same number as scikit-learn
    
    
    forest.learn(features_train, labels_train, numberOfTrees, 42);
    
    std::cout << "Learned " << forest.size() << " decision trees." << std::endl;
    
    // Predict using the learned forest
    const size_t shape[] = {features_test.shape(0), countUniqueLabels(labels_test)};
    std::cout << "Number of unique labels: " << shape[1] << std::endl;


    Marray<double> predictions(shape, shape+2);
    forest.predict(features_test, predictions);
    
    std::cout << "Predictions shape: " << predictions.shape(0) << " x " << predictions.shape(1) << std::endl;
    
    // Convert probabilities to class predictions
    std::vector<int> classPredictions = probabilitiesToPredictions(predictions);
    
    // Calculate accuracy
    double accuracy = calculateAccuracy(classPredictions, labels_test);
    std::cout << "Training accuracy: " << std::fixed << std::setprecision(4) << accuracy << std::endl;
    
    // Save results for comparison with scikit-learn
    try {
        savePredictionsToFile(classPredictions, path + "cpp_predictions.txt");
        saveProbabilitiesToFile(predictions, path + "cpp_probabilities.csv");
        std::cout << "Saved predictions to cpp_predictions.txt and cpp_probabilities.csv" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving files: " << e.what() << std::endl;
    }
    
    // Print sample predictions for manual inspection
    std::cout << "\nFirst 10 predictions:" << std::endl;
    std::cout << "Sample | Actual | Predicted | Probabilities" << std::endl;
    std::cout << "-------|--------|-----------|---------------" << std::endl;
    
    for (size_t i = 0; i < std::min(static_cast<size_t>(10), predictions.shape(0)); ++i) {
        std::cout << std::setw(6) << i << " | " 
                  << std::setw(6) << labels_test(i) << " | "
                  << std::setw(9) << classPredictions[i] << " | ";
        
        for (size_t j = 0; j < predictions.shape(1); ++j) {
            std::cout << std::fixed << std::setprecision(3) << predictions(i, j);
            if (j < predictions.shape(1) - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}