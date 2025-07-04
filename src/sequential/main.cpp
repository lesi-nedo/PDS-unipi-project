#include <vector>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <random>
#include <iostream>

#include "dt.h"
#include "marray.h"
#include "utils.h"
#include "seq_impl_config.h"
#include "general_config.h"

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
    std::string results_path = SEQ_RESULTS_PATH;
    if (results_path.empty()) {
        std::cerr << "Error: SEQ_RESULTS_PATH is not set. Please define it in the configuration file." << std::endl;
        return 1;
    }
    if (results_path[results_path.size() - 1] != '/')
        results_path += '/';

    std::cout << "Results will be saved to: " << results_path << std::endl;
    const std::vector<size_t> samples_perTree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> samples_perTree_test = DT_SAMPLES_PER_TREE_TEST; // Assuming same for test dataset

    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;

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