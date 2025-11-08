
#include "dt_mpi.h"
#include "marray.h"
#include "utils.h"
#include "mpi_impl_config.h"


int main(int argc, char* argv[]) {

    auto curr_path = std::filesystem::current_path();
    if (curr_path.filename() != "project") {
        std::cerr << "Error: Please run this program from the 'project' directory." << std::endl;
        return 1;
    }


    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        std::cerr << "This MPI implementation requires at least 2 processes." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::string test_dataset {argv[2]};
    std::string results_path {RESULTS_PATH_MPI};
    std::string results_file;
    std::ofstream performance_log;
    mpi_ml::DecisionForest<double, int, double> rf;

    const std::vector<size_t> samples_per_tree_test = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> samples_per_tree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;
    const int randomSeed = 42; // Fixed seed for reproducibility


    if (world_rank == 0 && results_path[results_path.size() - 1] != '/') {
        results_path += '/';
    }
    if (world_rank == 0) {
        const 
        std::filesystem::path path(results_file);

        if (samples_per_tree.size() != samples_per_tree_test.size()) {
            std::cerr << "Error: samples_per_tree and samples_per_tree_test must have the same number of elements." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        results_file = results_path + "performance_mpi.csv";
        std::cout << "Starting MPI Decision Forest training with " << world_size << " processes." << std::endl;
        std::cout << "Results will be saved to: " << results_file << std::endl;

        if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())){
            std::filesystem::create_directories(path.parent_path());
        }
        performance_log.open(results_file);
        saveTrainingDataHeader(performance_log);
        savePredictionDataHeader(performance_log);
    }

    for(int ind_sample = 0; ind_sample < samples_per_tree.size(); ++ind_sample){
        const auto& samples = samples_per_tree[ind_sample];
        const auto& samples_test = samples_per_tree_test[ind_sample];

        for (const auto& num_of_trees : tree_counts){

            if (world_rank == 0) {
                
                using namespace andres;

                rf.clear();
                
                std::cout << "\n--- Number of Trees: " << num_of_trees << " and  " << "Train Samples: " << samples << " ---" << std::endl;
                auto memory_before = getMemoryUsageMB();
        
                auto start_train = std::chrono::high_resolution_clock::now();
                rf.learnMaster(samples, num_of_trees, world_size, randomSeed);
                auto end_train = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> train_duration = end_train - start_train;
                auto memory_after = getMemoryUsageMB();
                double memory_usage = static_cast<double>(memory_after - memory_before);
                std::cout << "Learned " << rf.size() << " decision trees in " << train_duration.count() << " ms." << "Memory used: " << memory_usage << " MB." << std::endl;

                double trainingThroughput = samples / (train_duration.count() / 1000.0);
                auto training_data = std::make_tuple(num_of_trees, train_duration.count(), memory_usage, samples, trainingThroughput);
                addTrainRowToLog(performance_log, training_data);
            
                rf.terminateTraining(world_size);

                std::cout << "\nTraining phase completed. Starting prediction phase..." << std::endl;

                const auto [features_test, labels_test] = loadCSVToMarray<double>(std::string_view(test_dataset), ',', samples_test, andres::LastMajorOrder);
                std::cout << "Master starts prediction phase with " << features_test.size() << " samples." << std::endl;

                const size_t shapes[] = {features_test.shape(0), NUM_UNIQUE_LABELS};
                andres::Marray<double> probabilities(shapes, shapes + 2);
                std::fill(&probabilities(0), &probabilities(0) + probabilities.size(), 0.0);

                auto start_predict = std::chrono::high_resolution_clock::now();
                rf.predictMaster(samples_test, world_size, probabilities);
                auto end_predict = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> predict_duration = end_predict - start_predict;
                double predictionThroughput = samples_test / (predict_duration.count() / 1000.0);
                std::cout << "Prediction finished in " << predict_duration.count() << " ms." << std::endl;
                auto predicted_classes = probabilitiesToPredictions(probabilities);
                double accuracy = calculateAccuracy(predicted_classes, labels_test);
                auto [precision, recall, f1] = calculatePrecisionRecallF1(predicted_classes, labels_test);
                std::cout << "Accuracy: " << accuracy * 100.0 << "%" << std::endl << std::endl;
                auto prediction_data = std::make_tuple(predict_duration.count(), samples_test, probabilities.shape(1), predictionThroughput,  accuracy, f1, precision, recall);
                addPredictionRowToLog(performance_log, prediction_data);

                rf.terminatePrediction(world_size);
                std::cout << "Prediction phase completed." << std::endl;
                performance_log.flush();
                
            } else {
                
                std::string train_dataset = argv[1];

                rf.learnWorker(train_dataset, world_rank, randomSeed);
                rf.predictWorker(std::string_view(test_dataset), world_rank);

            }
        }
    }
    MPI_Finalize();
    return 0;
}