#include "utils.h"


size_t countUniqueLabels(const andres::Marray<int>& labels) {
    std::set<int> uniqueLabels;
    for (size_t i = 0; i < labels.shape(0); ++i) {
        uniqueLabels.insert(labels(i));
    }
    return uniqueLabels.size();
}



// Utility function to calculate accuracy
double calculateAccuracy(const std::vector<int>& predicted, const andres::View<int>& actual) {
    if (predicted.size() != actual.size()) {
        throw std::runtime_error("Predicted and actual labels size mismatch");
    }
    
    size_t correct = 0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        if (predicted[i] == actual(i)) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / predicted.size();
}

// Convert probability predictions to class predictions
std::vector<int> probabilitiesToPredictions(const andres::Marray<double>& probabilities) {
    std::vector<int> predictions(probabilities.shape(0));
    for (size_t i = 0; i < probabilities.shape(0); ++i) {
        int maxClass = 0;
        double maxProb = probabilities(i, 0);
        for (size_t j = 1; j < probabilities.shape(1); ++j) {
            if (probabilities(i, j) > maxProb) {
                maxProb = probabilities(i, j);
                maxClass = j;
            }
        }
        predictions[i] = maxClass;
    }
    return predictions;
}

// Save predictions to file for Python comparison
void savePredictionsToFile(const std::vector<int>& predictions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        file << predictions[i];
        if (i < predictions.size() - 1) file << "\n";
    }
    file.close();
}

// Save probabilities to file for Python comparison
void saveProbabilitiesToFile(const andres::Marray<double>& probabilities, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < probabilities.shape(0); ++i) {
        for (size_t j = 0; j < probabilities.shape(1); ++j) {
            file << probabilities(i, j);
            if (j < probabilities.shape(1) - 1) file << ",";
        }
        if (i < probabilities.shape(0) - 1) file << "\n";
    }
    file.close();
}


long getMemoryUsageMB(){
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024; // Convert to MB
}


std::tuple<double, double, double> calculatePrecisionRecallF1(
    const std::vector<int>& predictions, 
    const andres::Marray<int>& true_labels) {
    
    if (predictions.size() != true_labels.size()) {
        throw std::invalid_argument("Predictions and true labels must have the same size");
    }
    
    // Get unique classes
    std::set<int> unique_classes;
    for (int label : true_labels) {
        unique_classes.insert(label);
    }
    for (int pred : predictions) {
        unique_classes.insert(pred);
    }
    
    double total_precision = 0.0;
    double total_recall = 0.0;
    double total_f1 = 0.0;
    int valid_classes = 0;
    
    // Calculate metrics for each class
    for (int class_label : unique_classes) {
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (predictions[i] == class_label && true_labels(i) == class_label) {
                true_positives++;
            } else if (predictions[i] == class_label && true_labels(i) != class_label) {
                false_positives++;
            } else if (predictions[i] != class_label && true_labels(i) == class_label) {
                false_negatives++;
            }
        }
        
        // Calculate precision and recall for this class
        double precision = 0.0;
        double recall = 0.0;
        double f1 = 0.0;
        
        if (true_positives + false_positives > 0) {
            precision = static_cast<double>(true_positives) / (true_positives + false_positives);
        }
        
        if (true_positives + false_negatives > 0) {
            recall = static_cast<double>(true_positives) / (true_positives + false_negatives);
        }
        
        if (precision + recall > 0) {
            f1 = 2.0 * (precision * recall) / (precision + recall);
        }
        
        // Only include classes that have samples in true labels
        bool has_samples = std::find(true_labels.begin(), true_labels.end(), class_label) != true_labels.end();
        if (has_samples) {
            total_precision += precision;
            total_recall += recall;
            total_f1 += f1;
            valid_classes++;
        }
    }
    
    // Return macro-averaged metrics
    if (valid_classes > 0) {
        return std::make_tuple(
            total_precision / valid_classes,
            total_recall / valid_classes,
            total_f1 / valid_classes
        );
    } else {
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}