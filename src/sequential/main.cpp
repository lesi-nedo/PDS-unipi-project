#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <random>
#include <iostream>

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
    const std::string results_path = curr_path / "results" / "sequential_impl" / "";

    const std::vector<size_t> samples_perTree = {1000};
    const std::vector<size_t> samples_perTree_test = {100}; // Assuming same for test dataset

    const std::vector<size_t> tree_counts = {10, 50, 100, 150, 200};

    ml::DecisionForest<double, int, double> forest;

    try {
        run_test<double, int, double>(tree_counts, samples_perTree, samples_perTree_test,
                 std::string_view(train_dataset_path), 
                 std::string_view(test_dataset_path), 
                 results_path, forest, 42);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    
    return 0;
}