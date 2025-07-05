#pragma once
#ifndef FF_IMPL_CONFIG_HXX
#define FF_IMPL_CONFIG_HXX

// Experiment 1 configuration
#define FOR_CHUNK_SIZE 1
#define FOR_NUM_WORKERS 31
#define FOR_SPINWAIT false
#define FOR_SPINBARRIER false
#define FARM_NUM_WORKERS 2
#define MAP_SPINWAIT false
#define MAP_SPINBARRIER false
#define MAP_NUM_WORKERS 4
#define COMPLEXITY_THRESHOLD_FARM 50000
#define USE_PARALLEL_MAP false

// Experiment 2 configuration 
#define EN_NUM_WORKERS 6
#define WN_FIRST_NUM_WORKERS 2
#define WN_SECOND_NUM_WORKERS 24


// Prediction configuration
#define PEN_NUM_WORKERS 16 
#define MAX_PWN_NUM_WORKERS 2
#define CHUNK_SIZE_TO_PREDICT 100

// General configuration


#define RESULTS_PATH_EXP2 "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/fastflow_impl2"
#define RESULTS_PATH_EXP1 "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/fastflow_impl1"
#define CACHE_OPTIMIZATION false


#endif
