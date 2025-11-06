#include <iostream>
#include <filesystem>
#include <string>

#include "dt_omp.h"
#include "utils.h"
#include "general_config.h"
#include "omp_impl_config.h"
#include "marray.h"


int main(int argc, char* argv[]) {
    // Check if OpenMP is enabled
    #ifdef _OPENMP
        std::cout << "OpenMP is enabled." << std::endl;
    #else
        std::cerr << "OpenMP is not enabled." << std::endl;
        return 1;
    #endif
    if(argc < 3 || argc > 4) {
        std::cerr << "Usage: <tran_path> <test_path>" << std::endl;
        return 1;
    }

    std::string trainPath = argv[1];
    std::string testPath = argv[2];
    std::cout << "Training data path: " << trainPath << std::endl;
    std::cout << "Testing data path: " << testPath << std::endl;
    std::string currentPath = std::filesystem::current_path().string();


    auto curr_path = std::filesystem::current_path();
    if (curr_path.filename() != "project") {
        std::cerr << "Error: Please run this program from the 'project' directory." << std::endl;
        return 1;
    }
    std::string results_path = RESULTS_PATH_OMP;
    if (results_path[results_path.size() - 1] != '/') {
        results_path += '/';
    }
    std::cout << "Results will be saved to: " << results_path << std::endl;
    int randomSeed = 42;

    andres::omp_ml::DecisionForest<double, int, double> df;
    const std::vector<size_t> samplesPerTree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> samplesPerTreeTest = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> treeCounts = DT_TREE_COUNTS;

    try
    {
        run_test<double, int, double>(
            treeCounts, samplesPerTree, samplesPerTreeTest,
            std::string_view(trainPath), std::string_view(testPath), results_path, df, randomSeed
        );
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
}
    

