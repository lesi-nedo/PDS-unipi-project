#include "marray.h"
#include "utils.h"
#include "dt_ff_exp2.h"
#include "general_config.h"
#include "ff_impl_config.h"

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
    std::string results_path = RESULTS_PATH_EXP2;
    if (results_path[results_path.size() - 1] != '/') {
        results_path += '/';
    }
    std::cout << "Results will be saved to: " << results_path << std::endl;
    ff_ml_exp2::DecisionForest<double, int, double> rf;
    const int randomSeed = 42; // Fixed seed for reproducibility
    const std::vector<size_t> samples_perTree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> samples_perTree_test = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;
    const std::vector<std::tuple<int, int, int, int,int>> ff_exp2_threads = FF_EXP2_THREADS;
    const auto datasets_tuples = DATASETS_TUPLES;
    
    try {
        // Run comprehensive evaluation with thread variations
        std::cout << "\n=== Starting Comprehensive Performance Evaluation (Exp2) ===" << std::endl;
        std::cout << "Thread counts to test: ";
        for (const auto& [emitterWorkers, farmWorkers, treeWorkers, predWorkers1, predWorkers2] : ff_exp2_threads) 
            std::cout << emitterWorkers << " " << farmWorkers << " " << treeWorkers << " " << predWorkers1 << " " << predWorkers2 << " ";
        std::cout << std::endl;

        auto train_lambda = [randomSeed](auto& forest, const auto& features, const auto& labels, size_t numTrees, const std::vector<int>& workersConfigVec) {
            std::tuple<int, int, int> workersConfig;
            if (workersConfigVec.size() >= 3) {
                workersConfig = std::make_tuple(workersConfigVec[0], workersConfigVec[1], workersConfigVec[2]);
            } else {
                // Fallback logic if needed
                workersConfig = std::make_tuple(1, 1, 1);
            }
            forest.learnWithFFNetwork(features, labels, numTrees, workersConfig, randomSeed);
        };
        const int num_pred_workers = 2;
        for (const auto& [train_dataset, test_dataset, num_features] : datasets_tuples) {
            
            run_comprehensive_evaluation<double, int, double>(
                tree_counts, 
                samples_perTree, 
                samples_perTree_test,
                ff_exp2_threads,
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