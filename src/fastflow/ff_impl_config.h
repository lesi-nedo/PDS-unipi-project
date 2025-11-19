#pragma once
#ifndef FF_IMPL_CONFIG_HXX
#define FF_IMPL_CONFIG_HXX

// Experiment 1 configuration
#define FOR_CHUNK_SIZE -1
#define FOR_NUM_WORKERS 
#define FOR_SPINWAIT false
#define FOR_SPINBARRIER false
#define FARM_NUM_WORKERS 
#define MAP_SPINWAIT false
#define MAP_SPINBARRIER false
#define MAP_NUM_WORKERS 4
#define COMPLEXITY_THRESHOLD_FARM 
#define USE_PARALLEL_MAP false

// Experiment 2 configuration 
#define EN_NUM_WORKERS 12
#define WN_FIRST_NUM_WORKERS 2
#define WN_SECOND_NUM_WORKERS 2


// Prediction configuration
#define PEN_NUM_WORKERS 32 
#define MAX_PWN_NUM_WORKERS 1
#define CHUNK_SIZE_TO_PREDICT 50000
#define BATCH_SAMPLES_TO_PREDICT 32
#define PREFETCH_SAMPLES 24

// General configuration
#define RESULTS_PATH_EXP2 "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/fastflow_impl2"
#define RESULTS_PATH_EXP1 "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/fastflow_impl1"


#define FF_EXP1_THREADS {std::make_tuple(1,1,1), std::make_tuple(2,1,16), std::make_tuple(4,2,32), std::make_tuple(8,4,31), std::make_tuple(16,2,32), std::make_tuple(32,1,32)}
#define FF_EXP2_THREADS {std::make_tuple(1,1,1,1,1), std::make_tuple(2,1,1,16,16), std::make_tuple(4,2,2,12,12), std::make_tuple(8,4,4,10,22), std::make_tuple(16,2,8,6,26), std::make_tuple(32,1,16,6,16), std::make_tuple(11, 1, 20, 4, 8), std::make_tuple(8, 4, 20, 2,4), std::make_tuple(15, 2, 15, 2, 30)}



#endif
