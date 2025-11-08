#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <sys/time.h>

#include "dt_ff_exp1.h"
#include "utils.h"
#include "mpi.h"

#define TREES_TEST_SAMPLES_TAG 5
#define TREES_TRAIN_SAMPLES_TAG 4
#define TREES_DATA_TAG 3
#define TREES_SIZE_TAG 2
#define TREES_COUNT_TAG 1
#define TREES_ASSIGN_TAG 0
#define EOT_TAG 99 // End of Training
#define EOP_TAG 98 // End of Prediction


namespace mpi_ml {
    template<typename Feature, typename Label, typename Probability>
    class DecisionForest: public andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability> {
    
    private:
        std::tuple<std::streambuf*, std::streambuf*> redirectIOToLogFile(int world_rank, const std::string& log_filename, std::ofstream* log_stream, const std::string& process_name = "_worker_") {
            std::streambuf* original_cout_buf = std::cout.rdbuf();
            std::streambuf* original_cerr_buf = std::cerr.rdbuf();
            const std::string full_log_filename = log_filename + process_name + std::to_string(world_rank) + ".log";
            const std::filesystem::path log_path(full_log_filename);
            if (log_path.has_parent_path()) {
                std::filesystem::create_directories(log_path.parent_path());
            }
            *log_stream = std::ofstream(full_log_filename, std::ios::out | std::ios::trunc);
            if (!log_stream->is_open()) {
                std::cerr << "Error: Unable to open log file: " << full_log_filename << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::cout.rdbuf(log_stream->rdbuf());
            std::cerr.rdbuf(log_stream->rdbuf());

            return {original_cout_buf, original_cerr_buf};
        }
        void sendTrainedTrees(int world_rank, size_t startIndex, size_t endIndex) {
            if (startIndex >= endIndex || endIndex > andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability>::decisionTrees_.size()) {
                std::cerr << "Invalid tree index range to send from master to process " << world_rank << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            const unsigned long long numTrees = static_cast<unsigned long long>(endIndex - startIndex);
            MPI_Send(&numTrees, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_COUNT_TAG, MPI_COMM_WORLD);
            std::cout << "Sent " << numTrees << " trees to process " << world_rank << std::endl;
            std::vector<unsigned long long> treeSizes(numTrees);
            std::vector<std::string> serializedTrees(numTrees);
            unsigned long long totalDataSize = 0;
            for (size_t i = 0; i < numTrees; ++i) {
                std::ostringstream oss;
                andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability>::decisionTrees_[startIndex + i].serialize(oss);
                serializedTrees[i] = oss.str();
                treeSizes[i] = static_cast<unsigned long long>(serializedTrees[i].size());
                totalDataSize += treeSizes[i];
            }
            MPI_Send(treeSizes.data(), static_cast<int>(numTrees), MPI_UNSIGNED_LONG_LONG, world_rank, TREES_SIZE_TAG, MPI_COMM_WORLD);
            std::cout << "Sent tree sizes: " << totalDataSize << " to process " << world_rank << std::endl;
            std::vector<char> allTreeData;
            allTreeData.reserve(totalDataSize);
            for (const auto& treeData : serializedTrees) {
                allTreeData.insert(allTreeData.end(), treeData.begin(), treeData.end());
            }
            MPI_Send(allTreeData.data(), static_cast<int>(totalDataSize), MPI_CHAR, world_rank, TREES_DATA_TAG, MPI_COMM_WORLD);
            std::cout << "Sent all tree data to process " << world_rank << std::endl;
        }
        

        void receiveTrainedTrees(int world_rank){
            unsigned long long numTreesULL = 0;
            MPI_Recv(&numTreesULL, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_COUNT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process " << world_rank << " received number of trees to expect: " << numTreesULL << std::endl;
            size_t numTrees = static_cast<size_t>(numTreesULL);
            if (numTrees > 0) {
                std::vector<unsigned long long> treeSizes(numTrees);
                MPI_Recv(treeSizes.data(), static_cast<int>(numTrees), MPI_UNSIGNED_LONG_LONG, world_rank, TREES_SIZE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Process " << world_rank << " received tree sizes." << std::endl;
                size_t totalDataSize = 0;
                for (const auto& size : treeSizes) {
                    totalDataSize += size;
                }
                std::vector<char> allTreeData(totalDataSize);
                MPI_Recv(allTreeData.data(), static_cast<int>(totalDataSize), MPI_CHAR, world_rank, TREES_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Process " << world_rank << " received all tree data." << std::endl;
                size_t offset = 0;
                for (const auto& size : treeSizes) {
                    std::istringstream treeStream(std::string(allTreeData.data() + offset, size));
                    andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability>::decisionTrees_.emplace_back();
                    andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability>::decisionTrees_.back().deserialize(treeStream);
                    offset += size;
                }
                std::cout << "Process " << world_rank << " has received " << numTrees << " trees from master." << std::endl;
            } else {
                std::cerr << "Process " << world_rank << " received zero trees from master." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        void sendPredictionResults(int world_rank, const andres::Marray<Probability>& labelProbabilities) {
            if (labelProbabilities.dimension() != 2) {
                std::cerr << "Invalid labelProbabilities dimension to send from process " << world_rank << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            const unsigned long long rows = static_cast<unsigned long long>(labelProbabilities.shape(0));
            const unsigned long long cols = static_cast<unsigned long long>(labelProbabilities.shape(1));
            MPI_Send(&rows, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_COUNT_TAG, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_SIZE_TAG, MPI_COMM_WORLD);
            std::cout << "Process " << world_rank << " sent prediction result dimensions: " << rows << "x" << cols << std::endl;
            Probability* data_ptr = &labelProbabilities(0);
            MPI_Send(data_ptr, static_cast<int>(rows * cols), MPI_DOUBLE, world_rank, TREES_DATA_TAG, MPI_COMM_WORLD);
            std::cout << "Process " << world_rank << " sent prediction results." << std::endl;
        }
        std::pair<unsigned long long, unsigned long long> receivePredictionResultDimensions(int world_rank) {
            unsigned long long rows = 0;
            unsigned long long cols = 0;
            MPI_Recv(&rows, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_COUNT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&cols, 1, MPI_UNSIGNED_LONG_LONG, world_rank, TREES_SIZE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Process " << world_rank << " received prediction result dimensions: " << rows << "x" << cols << std::endl;
            return {rows, cols};
        }
        void genReceivePredictionResults(int world_rank, const unsigned long long rows, const unsigned long long cols, andres::Marray<Probability>& labelProbabilities) {
            
            
            if (rows != labelProbabilities.shape(0) || cols != labelProbabilities.shape(1)) {
                std::cerr << "Dimension mismatch in prediction results from process " << world_rank << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::vector<Probability> localProbabilities(rows * cols);

            MPI_Recv(localProbabilities.data(), static_cast<int>(rows * cols), MPI_DOUBLE, world_rank, TREES_DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Master received prediction results from process " << world_rank << std::endl;
            
            Probability* labelPtr = &labelProbabilities(0);
            const Probability* localPtr = localProbabilities.data();
            const size_t totalSize = rows * cols;
            for (size_t k = 0; k < totalSize; ++k) {
                labelPtr[k] += localPtr[k];
            }
        }
        
        void receivePredictionResults(int world_rank, andres::Marray<Probability>& labelProbabilities) {
            auto [rows, cols] = receivePredictionResultDimensions(world_rank);
            genReceivePredictionResults(world_rank, rows, cols, labelProbabilities);
        }
    public:
        using Base = andres::ff_ml_exp1::DecisionForest<Feature, Label, Probability>;
        using Base::Base;

        void learnMaster(
            const size_t num_train_samples,
            const size_t numberOfTrees,
            const int world_size,
            const int randomSeed = 0
        ) {

            std::ofstream log_file;
            auto [coutBuf, cerrBuf] = redirectIOToLogFile(0, "logs/mpi/", &log_file, "_master_");
            
            // Master process
            size_t treesPerWorker = numberOfTrees / (world_size - 1);
            size_t remainingTrees = numberOfTrees % (world_size - 1);
            // Distribute work to worker processes
            for (int worker = 1; worker < world_size; ++worker) {
                const unsigned long long num_train_samples_ull = static_cast<unsigned long long>(num_train_samples);
                MPI_Send(&num_train_samples_ull, 1, MPI_UNSIGNED_LONG_LONG, worker, TREES_TRAIN_SAMPLES_TAG, MPI_COMM_WORLD);
                std::cout << "Sent number of training samples: " << num_train_samples << " to process " << worker << std::endl;
                size_t treesToAssign = treesPerWorker + (worker <= remainingTrees ? 1 : 0);
                const unsigned long long treesToAssignULL = static_cast<unsigned long long>(treesToAssign);
                MPI_Send(&treesToAssignULL, 1, MPI_UNSIGNED_LONG_LONG, worker, TREES_ASSIGN_TAG, MPI_COMM_WORLD);
                std::cout << "Assigned " << treesToAssign << " trees to process " << worker << std::endl;
            }
            
            // Collect trained trees from workers
            for (int worker = 1; worker < world_size; ++worker) {
                receiveTrainedTrees(worker);
                std::cout << "Master received trained trees from process " << worker << std::endl;
            }
            std::cout << "Master has collected all trained trees. Total trees in forest: " << Base::size() << std::endl;
            // Restore std::cout and std::cerr
            std::cout.rdbuf(coutBuf);
            std::cerr.rdbuf(cerrBuf);
            log_file.close();
        }
        void predictMaster(
            const size_t num_test_samples,
            const int world_size,
            andres::Marray<Probability>& labelProbabilities
        ) {
            // Master process
            std::ofstream log_file;
            auto [coutBuf, cerrBuf] = redirectIOToLogFile(0, "logs/mpi/", &log_file, "predict_master_");
            
            // Distribute work to worker processes
            for (int worker = 1; worker < world_size; ++worker) {
                const unsigned long long num_test_samples_ull = static_cast<unsigned long long>(num_test_samples);
                MPI_Send(&num_test_samples_ull, 1, MPI_UNSIGNED_LONG_LONG, worker, TREES_TEST_SAMPLES_TAG, MPI_COMM_WORLD);
                std::cout << "Sent number of test samples: " << num_test_samples << " to process " << worker << std::endl;
            }
            auto size_dt = Base::size();
            const size_t trees_per_worker = size_dt / (world_size - 1);
            const size_t remaining_trees = size_dt % (world_size - 1);
            for (int worker = 1; worker < world_size; ++worker) {
                size_t treesToAssign = trees_per_worker + (worker <= remaining_trees ? 1 : 0);
                if (treesToAssign > 0) {
                    const size_t startIndex = (worker - 1) * trees_per_worker + std::min(static_cast<size_t>(worker - 1), remaining_trees);
                    const size_t endIndex = startIndex + treesToAssign;
                    sendTrainedTrees(worker, startIndex, endIndex);
                    std::cout << "Sent " << treesToAssign << " trees to process " << worker << " for prediction." << std::endl;
                } else {
                    std::cerr << "No trees to assign to process " << worker << " for prediction." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            
            for (int worker = 1; worker < world_size; ++worker) {
                receivePredictionResults(worker, labelProbabilities);
                std::cout << "Master received prediction results from process " << worker << std::endl;
            }
            // Restore std::cout and std::cerr
            std::cout.rdbuf(coutBuf);
            std::cerr.rdbuf(cerrBuf);
            log_file.close();
        }
        
        void terminateTraining(int world_size) {
            unsigned int eosFlag = EOT_TAG;
            for (int worker = 1; worker < world_size; ++worker) {
                MPI_Send(&eosFlag, 1, MPI_UNSIGNED, worker, EOT_TAG, MPI_COMM_WORLD);
            }
        }
        void terminatePrediction(int world_size) {
            unsigned int eopFlag = EOP_TAG;
            for (int worker = 1; worker < world_size; ++worker) {
                MPI_Send(&eopFlag, 1, MPI_UNSIGNED, worker, EOP_TAG, MPI_COMM_WORLD);
            }
        }
            

        void learnWorker(
            const std::string_view train_dataset_path,
            int world_rank,
            const int randomSeed = 0
        ) {
            // starting time measurement
            struct timeval start, end;
            gettimeofday(&start, NULL);
            const int random_seed = randomSeed + world_rank; // Different seed per worker
            
            std::ofstream log_file;
            auto [coutBuf, cerrBuf] = redirectIOToLogFile(world_rank, "logs/mpi/", &log_file, "_worker_");
            bool eot_received = false;

            while (!eot_received) {

                unsigned long long num_train_samples_ull = 0;
                unsigned long long treesToTrainULL = 0;
                while(true){
                    MPI_Status status;
                    MPI_Recv(&num_train_samples_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    if (status.MPI_TAG == EOT_TAG) {
                        eot_received = true;
                        break;
                    } else if (status.MPI_TAG == TREES_TRAIN_SAMPLES_TAG) {
                        MPI_Recv(&treesToTrainULL, 1, MPI_UNSIGNED_LONG_LONG, 0, TREES_ASSIGN_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        break;
                    } else {
                        std::cerr << "Worker " << world_rank << " received unexpected MPI tag: " << status.MPI_TAG << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                    
                    
                }
                std::cout << "Worker " << world_rank << " will train " << treesToTrainULL << " trees with " << num_train_samples_ull << " samples each." << std::endl;
                const size_t treesToTrain = static_cast<size_t>(treesToTrainULL);

                if (treesToTrain > 0 && !eot_received) {
                    Base::clear();
                    const auto [features, labels] = loadCSVToMarray<Feature>(train_dataset_path, ',', static_cast<size_t>(num_train_samples_ull), andres::LastMajorOrder);
                    Base::learn(features, labels, treesToTrain, random_seed);
                    std::cout << "Worker " << world_rank << " finished training " << treesToTrain << " trees." << std::endl;
                    sendTrainedTrees(0, Base::size() - treesToTrain, Base::size());
                    std::cout << "Worker " << world_rank << " sent trained trees back to master." << std::endl;
                } else if (treesToTrain == 0 && !eot_received) {
                    std::cerr << "Worker " << world_rank << " received zero trees to train." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            gettimeofday(&end, NULL);
            double elapsed = ff::diffmsec(end, start);
            std::cout << "Worker "<< world_rank << " finished in " << elapsed << " milliseconds." << std::endl;
            // Restore std::cout and std::cerr
            std::cout.rdbuf(coutBuf);
            std::cerr.rdbuf(cerrBuf);
            log_file.close();

        }

        void predictWorker(
            const std::string_view test_dataset_path,
            int world_rank
        ){
            // starting time measurement
            struct timeval start, end;
            gettimeofday(&start, NULL);

            std::ofstream log_file;
            auto [coutBuf, cerrBuf] = redirectIOToLogFile(world_rank, "logs/mpi/", &log_file, "predict_worker_");
            bool eop_received = false;

            while (!eop_received) {

                unsigned long long num_test_samples_ull = 0;
                while(true){
                    MPI_Status status;
                    MPI_Recv(&num_test_samples_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    if (status.MPI_TAG == EOP_TAG) {
                        eop_received = true;
                        std::cout << "Worker " << world_rank << " received end of prediction signal." << std::endl;
                    } else if (status.MPI_TAG != TREES_TEST_SAMPLES_TAG) {
                        std::cerr << "Worker " << world_rank << " received unexpected MPI tag: " << status.MPI_TAG << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    break;
                }
                
                if (num_test_samples_ull > 0 && !eop_received) {
                    std::cout << "Worker " << world_rank << " predicting with " << Base::size() << " trees." << std::endl;
                    Base::clear();
                    receiveTrainedTrees(0);
                    const auto [features, labels] = loadCSVToMarray<Feature>(test_dataset_path, ',', static_cast<size_t>(num_test_samples_ull), andres::LastMajorOrder);
                    const size_t shapes[] = {features.shape(0), NUM_UNIQUE_LABELS};
                    andres::Marray<Probability> labelProbabilities(shapes, shapes + 2);
                    std::cout << "Worker " << world_rank << " loaded test data, starting predictions." << std::endl;
                    Base::predict(features, labelProbabilities);
                    std::cout << "Worker " << world_rank << " finished predictions." << std::endl;
                    sendPredictionResults(0, labelProbabilities);
                    std::cout << "Worker " << world_rank << " sent prediction results back to master." << std::endl;
                } else if (num_test_samples_ull == 0 && !eop_received) {
                    std::cerr << "Worker " << world_rank << " received zero samples to predict." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

            }
            gettimeofday(&end, NULL);
            double elapsed = ff::diffmsec(end, start);
            std::cout << "Worker "<< world_rank << " finished in " << elapsed << " milliseconds." << std::endl;
            // Restore std::cout and std::cerr
            std::cout.rdbuf(coutBuf);
            std::cerr.rdbuf(cerrBuf);
            log_file.close();
        }

        Base getDecisionForest() const {
            return *this;
        }
    };

    
}