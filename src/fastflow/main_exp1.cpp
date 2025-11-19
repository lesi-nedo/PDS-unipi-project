#include "marray.h"
#include "utils.h"
#include "dt_ff_exp1.h"
#include "general_config.h"
#include "ff_impl_config.h"


int main(int argc, char* argv[]) {

    using namespace andres;


    auto curr_path = std::filesystem::current_path();
    
    if (curr_path.filename() != "project") {
        std::cerr << "Error: Please run this program from the 'project' directory." << std::endl;
        return 1;
    }
    std::string results_path = RESULTS_PATH_EXP1;
    if (results_path[results_path.size() - 1] != '/') {
        results_path += '/';
    }
    std::cout << "Results will be saved to: " << results_path << std::endl;
    ff_ml_exp1::DecisionForest<double, int, double> rf;
    const int randomSeed = 42; // Fixed seed for reproducibility
    const std::vector<size_t> samples_perTree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> samples_perTree_test = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;
    const std::vector<std::tuple<int, int, int>> ff_exp1_threads = FF_EXP1_THREADS;
    const auto datasets_tuples = DATASETS_TUPLES;
    
    // Thread count variations for comprehensive evaluation

    try {
        // Run comprehensive evaluation with thread variations
        std::cout << "\n=== Starting Comprehensive Performance Evaluation (Exp1) ===" << std::endl;
        std::cout << "Thread counts to test: ";
        for (const auto& [farmWorkers, treeWorkers, predWorkers] : ff_exp1_threads) 
            std::cout << farmWorkers << " " << treeWorkers << " " << predWorkers << " ";
        std::cout << std::endl;
        std::cout << "Number of workers: ";
        for (const auto& [farmWorkers, treeWorkers, predWorkers] : ff_exp1_threads) 
            std::cout << "Farms: " << farmWorkers << " Trees: " << treeWorkers << " Pred:  " << predWorkers << std::endl;

        auto train_lambda = [randomSeed](auto& forest, const auto& features, const auto& labels, size_t numTrees, const std::vector<int>& workersConfigVec) {
            std::tuple<int, int> workersConfig;
            if (workersConfigVec.size() >= 2) {
                workersConfig = std::make_tuple(workersConfigVec[0], workersConfigVec[1]);
            } else if (workersConfigVec.size() == 1) {
                workersConfig = std::make_tuple(workersConfigVec[0], 0);
            } else {
                workersConfig = std::make_tuple(1, 0);
            }
            forest.learn(features, labels, numTrees, workersConfig, randomSeed);
        };
        const int num_pred_workers = 1;
        for (const auto& [train_dataset, test_dataset, num_features] : datasets_tuples) {

            run_comprehensive_evaluation<double, int, double, ff_ml_exp1::DecisionForest<double, int, double>, decltype(train_lambda), int, int, int>(
                tree_counts, 
                samples_perTree, 
                samples_perTree_test,
                ff_exp1_threads,
                num_pred_workers,
                std::string_view(train_dataset), 
                std::string_view(test_dataset), 
                results_path, 
                rf,
                train_lambda,
                num_features,
                randomSeed
            );
        }
        
        
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}