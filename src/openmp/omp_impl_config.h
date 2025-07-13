#pragma once
#ifndef OMP_IMPL_CONFIG_HXX
#define OMP_IMPL_CONFIG_HXX

#define PREFETCH_SAMPLES 26
#define GENERAL_NUM_OMP_THREADS 32
#define USE_NESTED_PARALLELISM true
#define USE_DYNAMIC_THREAD_SCHEDULING true
#define REC_NUM_OMP_THREADS 
#define NODE_PRAL_NUM_OMP_THREADS 
#define CHUNK_SIZE 1

#define SCHEDULE_TYPE "dynamic"
#define RESULTS_PATH_OMP "/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project/results/openmp"


#define PRED_GENERAL_NUM_OMP_THREADS 32
#define PRED_USE_NESTED_PARALLELISM true
#define PRED_USE_DYNAMIC_THREAD_SCHEDULING true
#define PRED_CHUNK_SIZE 4
#define PRED_SCHEDULE_TYPE "dynamic"

#define NUM_OF_SAMPLES_BEFORE_PARALLEL 10000

#endif
