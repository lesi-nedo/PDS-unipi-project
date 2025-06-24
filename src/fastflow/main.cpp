#include "../../include/sequential/marray.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <train_dataset_path> <test_dataset_path>" << std::endl;
        return 1;
    }

    using namespace andres;

    std::string train_dataset = argv[1];
    std::string test_dataset = argv[2];
    auto [features_train, labels_train] = loadCSVToMarray<double>(train_dataset,',', 1000);
    auto [features_test, labels_test] = loadCSVToMarray<double>(test_dataset, ',', 1000);

    std::cout << "Train Features shape: " << features_train.shape(0) << " x " << features_train.shape(1) << std::endl;
    std::cout << "Train Labels shape: " << labels_train.shape(0) << std::endl;
    std::cout << "Test Features shape: " << features_test.shape(0) << " x " << features_test.shape(1) << std::endl;
    std::cout << "Test Labels shape: " << labels_test.shape(0) << std::endl;
}