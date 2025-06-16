#include "utils.h"


size_t countUniqueLabels(const andres::Marray<int>& labels) {
    std::set<int> uniqueLabels;
    for (size_t i = 0; i < labels.shape(0); ++i) {
        uniqueLabels.insert(labels(i));
    }
    return uniqueLabels.size();
}