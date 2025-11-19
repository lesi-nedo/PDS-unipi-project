#pragma once
#ifndef RF_FASTFLOW_CONFIG_HXX
#define RF_FASTFLOW_CONFIG_HXX

#define DT_TREE_COUNTS {36, 50, 100, 150, 200}
#define DT_SAMPLES_PER_TREE_TEST {1000}
#define DT_SAMPLES_PER_TREE {2000}
#define NUM_UNIQUE_LABELS 2
#define MIN_THREAD_COUNT 
#define MAX_THREAD_COUNT 
#define THREAD_COUNT_STEPS 
#define ENABLE_DYNAMIC_THREAD_ADJUSTMENT true
#define DATASETS_TUPLES {std::make_tuple("./data/susy/train_susy.csv", "./data/susy/test_susy.csv", 2)}

#endif
