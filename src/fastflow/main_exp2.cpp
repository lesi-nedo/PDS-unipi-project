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

    try {
        run_test_ff_network<double, int, double>(tree_counts, samples_perTree, samples_perTree_test, 
                 std::string_view(train_dataset), 
                 std::string_view(test_dataset), 
                 results_path, rf, randomSeed);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    

}