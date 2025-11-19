
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

    if (argc != 3) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <train_dataset_path> <test_dataset_path>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string train_dataset {argv[1]};
    std::string test_dataset {argv[2]};
    std::string results_path {RESULTS_PATH_MPI};
    mpi_ml::DecisionForest<double, int, double> forest;
    
    const std::vector<size_t> samples_per_tree_test = DT_SAMPLES_PER_TREE_TEST;
    const std::vector<size_t> samples_per_tree = DT_SAMPLES_PER_TREE;
    const std::vector<size_t> tree_counts = DT_TREE_COUNTS;
    const std::vector<std::tuple<int, int>> thread_counts = FF_THREADS;
    const auto datasets_tuples = std::vector<std::tuple<std::string, std::string, size_t>>{ DATASETS_TUPLES };

    const int randomSeed = 42; // Fixed seed for reproducibility

    if (world_rank == 0){
        if (results_path[results_path.size() - 1] != '/') {
            results_path += '/';
        }
        std::filesystem::create_directories(results_path);
        const auto file_name = "performance_nodes_" + std::to_string(world_size) + ".csv";
        std::ofstream performance_log(results_path + file_name);

        std::unordered_map<std::string, double> baseline_train_times;
        std::unordered_map<std::string, double> baseline_pred_times;

        performance_log << "NumNodes,MPIProcesses,ThreadsPerProcess,TotalWorkers,"
                        << "NumTrees,TrainSamples,TestSamples,"
                        << "TrainingTime_ms,PredictionTime_ms,TotalTime_ms,"
                        << "Accuracy,F1Score,Precision,Recall,"
                        << "TrainingStrongScalingEfficiency,TrainingWeakScalingEfficiency,"
                        << "PredictionStrongScalingEfficiency,PredictionWeakScalingEfficiency,"
                        << "TrainingSpeedup,PredictionSpeedup,"
                        << "TrainingCommunicationOverhead_ms,PredictionCommunicationOverhead_ms,MemoryUsage_MB\n";

        std::cout << "\n=== Starting MPI Performance Evaluation ===" << std::endl;

        try {
            for (const auto& [train_path, test_path, unique_labels] : datasets_tuples){
                if(!std::filesystem::exists(train_path) || !std::filesystem::exists(test_path)) {
                    std::cerr << "Error: Dataset files not found: " 
                              << train_path << " or " << test_path << std::endl;
                    throw std::runtime_error("Train and test dataset files not found.");
                }

                for (const auto& [farmWorkers, workersPerTree]: thread_counts){
                    const auto total_threads = farmWorkers * workersPerTree;
                    const auto total_workers = world_size * total_threads;

                    std::cout << "\n=== MPI Configuration: Processes=" << world_size
                            << ", ThreadsPerProcess=" << total_threads
                            << ", TotalWorkers=" << total_workers << "===" << std::endl;

                    for(size_t idx = 0; idx < samples_per_tree.size(); ++idx) {
                        const auto samples_train = samples_per_tree[idx];
                        const auto samples_test = samples_per_tree_test[idx];

                        for (const auto& numberOfTrees : tree_counts) {
                            const std::string config_key = std::to_string(numberOfTrees) + "_" 
                                                        + std::to_string(samples_train);

                                    
                            std::cout << "\n--- Testing Configuration: Trees=" << numberOfTrees
                                    << ", TrainSamples=" << samples_train
                                    << ", TestSamples=" << samples_test << " ---" << std::endl;
                        
                            long memory_before = getMemoryUsageMB();
                            auto start_train = std::chrono::high_resolution_clock::now();
                            forest.learnMaster(samples_train, numberOfTrees, world_size, randomSeed);
                            auto end_train = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double, std::milli> train_duration = end_train - start_train;
                            long memory_after_train = getMemoryUsageMB();

                            forest.terminateTraining(world_size);

                            const size_t shape[] = {samples_test, unique_labels};
                            andres::Marray<double> probabilities(shape, shape + 2);
                            std::fill(&probabilities(0), &probabilities(0) + probabilities.size(), 0.0);

                            auto start_predict = std::chrono::high_resolution_clock::now();
                            forest.predictMaster(samples_test, world_size, probabilities);
                            auto end_predict = std::chrono::high_resolution_clock::now();
                            std::chrono::duration<double, std::milli> pred_duration = end_predict - start_predict;


                            const auto memory_after_predict = getMemoryUsageMB();
                            const auto memory_usage = memory_after_predict - memory_after_train;
                            forest.terminatePrediction(world_size);

                            auto [features_test, labels_test] = loadCSVToMarray<double>(
                                test_path, ',', samples_test, andres::LastMajorOrder
                            );

                            auto [accuracy, f1, precision, recall] = calculateMetrics(
                                probabilities, labels_test
                            );

                            const double total_time = train_duration.count() + pred_duration.count();
                            double train_speedup = 1.0, pred_speedup = 1.0;
                            double train_strong_efficiency = 1.0, pred_strong_efficiency = 1.0;
                            double train_weak_efficiency = 1.0, pred_weak_efficiency = 1.0;

                            if (threads == thread_counts[0]) {
                                baseline_train_times[config_key] = train_duration.count();
                                baseline_pred_times[config_key] = pred_duration.count();
                            } else {
                                train_speedup = baseline_train_times[config_key] / train_duration.count();
                                pred_speedup = baseline_pred_times[config_key] / pred_duration.count();
                                train_strong_efficiency = train_speedup / static_cast<double>(total_workers);
                                pred_strong_efficiency = pred_speedup / static_cast<double>(total_workers);
                                train_weak_efficiency = baseline_train_times[config_key] / train_duration.count();
                                pred_weak_efficiency = baseline_pred_times[config_key] / pred_duration.count();
                                


                            }
                            double train_est_comm_overhead = train_duration.count() - 
                                               (baseline_train_times[config_key] / total_workers);

                            double pred_est_comm_overhead = pred_duration.count() - 
                                               (baseline_pred_times[config_key] / total_workers);

                            // Log results
                            performance_log << world_size << ","
                                            << world_size << ","
                                            << threads << ","
                                            << total_workers << ","
                                            << numberOfTrees << ","
                                            << samples_train << ","
                                            << samples_test << ","
                                            << train_duration.count() << ","
                                            << pred_duration.count() << ","
                                            << total_time << ","
                                            << accuracy << ","
                                            << f1 << ","
                                            << precision << ","
                                            << recall << ","
                                            << train_strong_efficiency << ","
                                            << train_weak_efficiency << ","
                                            << pred_strong_efficiency << ","
                                            << pred_weak_efficiency << ","
                                            << train_speedup << ","
                                            << pred_speedup << ","
                                            << train_est_comm_overhead << ","
                                            << pred_est_comm_overhead << ","
                                            << memory_usage << "\n";
                            performance_log.flush();
                            std::cout << "Training Time: " << train_duration.count() << " ms" << std::endl;
                            std::cout << "Prediction Time: " << pred_duration.count() << " ms" << std::endl;
                            std::cout << "Training Speedup: " << train_speedup << "x" << std::endl;
                            std::cout << "Prediction Speedup: " << pred_speedup << "x" << std::endl;
                            std::cout << "Training Strong Scaling Efficiency: " << (train_strong_efficiency * 100) << "%" << std::endl;
                            std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;

                            forest.clear();
                        }
                    }
                }
            }

        } catch (const std::runtime_error& e) {
            std::cerr << "Error during MPI evaluation: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    } else {
       for (const auto& [train_path, test_path, unique_labels] : datasets_tuples){
            for (const auto& threads: thread_counts){
                for(size_t idx = 0; idx < samples_per_tree.size(); ++idx) {

                    for (const auto& numberOfTrees : tree_counts) {
                        forest.learnWorker(
                            std::string_view(train_path),
                            world_rank
                        );
                        forest.predictWorker(
                            std::string_view(test_path),
                            world_rank
                        );
                    }
                }
            }
       }
    }

    MPI_Finalize();
    return 0;
}