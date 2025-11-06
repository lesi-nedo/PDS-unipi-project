
#include "dt_mpi.h"
#include "marray.h"
#include "utils.h"
#include "general_config.h"
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
    const std::vector<size_t> samples_perTree_test = DT_SAMPLES_PER_TREE_TEST;
    const int randomSeed = 42; // Fixed seed for reproducibility

    if (world_rank == 0) {
        
        using namespace andres;

        std::cout << "Starting MPI Decision Forest training with " << world_size << " processes." << std::endl;

        mpi_ml::DecisionForest<double, int, double> rf;


        
        
        if (results_path[results_path.size() - 1] != '/') {
            results_path += '/';
        }
        std::cout << "Results will be saved to: " << results_path << std::endl;
        
        
        const std::vector<size_t> samples_perTree = DT_SAMPLES_PER_TREE;
        const std::vector<size_t> tree_counts = DT_TREE_COUNTS;
        std::vector<mpi_ml::DecisionForest<double, int, double>> rf_trained;
        const std::string results_file = results_path + "train_performance_mpi.csv";
       
        try {
            run_training_mpi<double, int, double>(
                tree_counts, samples_perTree,
                results_file,
                world_size, randomSeed, rf_trained
            );

        } catch (const std::runtime_error& e) {
            std::cerr << "Error during testing: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rf.terminateTraining(world_size);
        std::cout << "\nTraining phase completed. Starting prediction phase..." << std::endl;

        // const std::string prediction_results_file = results_path + "performance_mpi_pred_with_master.csv";
        const std::string prediction_results_file = results_path + "performance_mpi.csv";
        std::ifstream performance_log(results_file);
        std::vector<std::string> lines;
        std::string line;
        std::ofstream prediction_log_out(prediction_results_file);


        while (std::getline(performance_log, line)) {
            lines.push_back(line);
        }
        performance_log.close();
        prediction_log_out << lines[0];
        savePredictionDataHeader(prediction_log_out);
        

        for (const auto& sample_test: samples_perTree_test) {
            const auto [features_test, labels_test] = loadCSVToMarray<double>(std::string_view(test_dataset), ',', sample_test, andres::LastMajorOrder);
            std::cout << "Master starts prediction phase with " << features_test.size() << " samples." << std::endl;
            
            
            const auto unique_labels = countUniqueLabels(labels_test);
            const size_t shapes[] = {features_test.shape(0), unique_labels};
            
            
            int count = 1;
            for (auto& forest : rf_trained){
                andres::Marray<double> probabilities(shapes, shapes + 2);
                prediction_log_out << lines[count++];
                std::fill(&probabilities(0), &probabilities(0) + probabilities.size(), 0.0);

                try {
                    //only master predicts
                    // run_prediction(  
                    //     features_test,
                    //     labels_test,
                    //     forest.getDecisionForest(),
                    //     probabilities,
                    //     prediction_log_out   
                    // );
                    // Distributed prediction across the workers
                    run_prediction_mpi<double, int, double>(
                        labels_test,
                        forest,
                        probabilities,
                        prediction_log_out,
                        sample_test,
                        world_size
                    );
                } catch (const std::runtime_error& e) {
                    std::cerr << "Error during prediction: " << e.what() << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }


        }
        rf.terminatePrediction(world_size);
        std::cout << "Prediction phase completed." << std::endl;
        prediction_log_out.flush();
        prediction_log_out.close();
        
    } else {
        
        std::string train_dataset = argv[1];
        
        mpi_ml::DecisionForest<double, int, double> rf_worker;
        rf_worker.learnWorker(train_dataset, world_rank, randomSeed);
        rf_worker.predictWorker(std::string_view(test_dataset), world_rank);
    
    }
    MPI_Finalize();
    return 0;
}