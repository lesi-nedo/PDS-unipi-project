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
    std::string results_path = "./results/ff_sklearn/";
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
        
        }
    }