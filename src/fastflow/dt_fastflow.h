/// \mainpage
/// andres::ml::DecisionTrees: A Fast C++ Implementation of Random Forests
///
/// \section section_abstract Short Description
/// The header file implements random forests as described in the article:
///
/// Leo Breiman. Random Forests. Machine Learning 45(1):5-32, 2001.
/// http://dx.doi.org/10.1023%2FA%3A1010933404324
/// 
/// \section section_license License
///
/// Copyright (c) 2013 by Steffen Kirchhoff and Bjoern Andres.
///
/// This software was developed by Steffen Kirchhoff and Bjoern Andres.
/// Enquiries shall be directed to bjoern@andres.sc.
///
/// All advertising materials mentioning features or use of this software must
/// display the following acknowledgement: ``This product includes andres::ml 
/// Decision Trees developed by Steffen Kirchhoff and Bjoern Andres. Please 
/// direct enquiries concerning andres::ml Decision Trees to bjoern@andres.sc''.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions are met:
///
/// - Redistributions of source code must retain the above copyright notice,
/// this list of conditions and the following disclaimer.
/// - Redistributions in binary form must reproduce the above copyright notice,
/// this list of conditions and the following disclaimer in the documentation
/// and/or other materials provided with the distribution.
/// - All advertising materials mentioning features or use of this software must
/// display the following acknowledgement: ``This product includes andres::ml 
/// Decision Trees developed by Steffen Kirchhoff and Bjoern Andres. Please 
/// direct enquiries concerning andres::ml Decision Trees to bjoern@andres.sc''.
/// - The names of the authors must not be used to endorse or promote products
/// derived from this software without specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR IMPLIED
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
/// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
/// EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
/// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
/// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
/// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
/// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
/// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/// 
#pragma once
#ifndef ANDRES_ML_DECISION_FOREST_HXX
#define ANDRES_ML_DECISION_FOREST_HXX

#include <stdexcept>
#include <random>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <map>
#include <tuple>
#include <immintrin.h>


#include "ff/ff.hpp"
#include "ff/parallel_for.hpp"
#include "ff/farm.hpp"

#include "./ff_impl_config.h"

#include "marray.h"

/// The public API.
namespace andres {
    
/// Machine Learning.
namespace ff_ml {

class OptimizedGiniCalculator {
public:
    // Calculates sum_{i<j} counts[i]*counts[j] using the identity:
    // sum_{i<j} c_i*c_j = ( (sum c_i)^2 - sum(c_i^2) ) / 2
    // This is much faster than a nested loop.
    static size_t computeDistinctPairs(const std::vector<size_t>& labelCounts, size_t totalSamples) {
        if (totalSamples < 2) {
            return 0;
        }
        
        // 1. Calculate sum of squares (vectorized with AVX2)
        // We use doubles for floating point SIMD instructions. This is safe as long
        // as counts do not exceed 2^53, which is highly unlikely for sample counts.
        __m256d sum_sq_vec = _mm256_setzero_pd();
        size_t i = 0;
        
        // Process 4 counts (as doubles) at a time
        for (; i + 4 <= labelCounts.size(); i += 4) {
            // Manual conversion from size_t to double for AVX2
            double vals[4] = {
                static_cast<double>(labelCounts[i]), static_cast<double>(labelCounts[i+1]),
                static_cast<double>(labelCounts[i+2]), static_cast<double>(labelCounts[i+3])
            };
            __m256d counts_pd = _mm256_loadu_pd(vals);

            // Fused-Multiply-Add can be slightly faster if available and applicable
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(counts_pd, counts_pd));
        }

        // Horizontal sum of the vector's partial sums
        alignas(32) double sum_sq_parts[4];
        _mm256_store_pd(sum_sq_parts, sum_sq_vec);
        double sum_of_squares = sum_sq_parts[0] + sum_sq_parts[1] + sum_sq_parts[2] + sum_sq_parts[3];

        // Handle remaining elements not processed by SIMD
        for (; i < labelCounts.size(); ++i) {
            sum_of_squares += static_cast<double>(labelCounts[i]) * static_cast<double>(labelCounts[i]);
        }

        // 2. Use the identity to find the number of distinct pairs
        double totalSamples_d = static_cast<double>(totalSamples);
        double total_pairs_d = ((totalSamples_d * totalSamples_d) - sum_of_squares) / 2.0;
        
        // Add 0.5 for proper rounding before casting to size_t
        return static_cast<size_t>(total_pairs_d + 0.5);
    }
};

/// A node in a decision tree.
template<class FEATURE, class LABEL>
class DecisionNode {
public:
    typedef FEATURE Feature;
    typedef LABEL Label;

    DecisionNode();
    bool& isLeaf();
    size_t& featureIndex();
    Feature& threshold();
    size_t& childNodeIndex(const size_t);
    Label& label();
    void deserialize(std::istream&);

    bool isLeaf() const;
    size_t featureIndex() const;
    Feature threshold() const;
    size_t childNodeIndex(const size_t) const;
    Label label() const;
    size_t learn(const andres::View<Feature>&, const andres::View<Label>&, 
            std::vector<size_t>&, const size_t, const size_t, const int = 0);   
    void serialize(std::ostream&) const;

private:
    struct ComparisonByFeature {
        typedef size_t first_argument_type;
        typedef size_t second_argument_type;
        typedef bool result_type;

        ComparisonByFeature(
            const andres::View<Feature>& features,
            const size_t featureIndex
        )
            :   features_(features),
                featureIndex_(featureIndex)
            {
                assert(featureIndex < features.shape(1));
            }
        bool operator()(const size_t j, const size_t k) const 
            { 
                assert(j < features_.shape(0));
                assert(k < features_.shape(0));
                return features_(j, featureIndex_) < features_(k, featureIndex_);
            }

        const andres::View<Feature>& features_;
        const size_t featureIndex_;
    };

    struct SortItem {
        Feature featureValue;
        size_t sampleIndex;
        bool operator<(const SortItem& other) const {
                return featureValue < other.featureValue;
        }
    };

    template<class RandomEngine>
        void sampleSubsetWithoutReplacement(const size_t, const size_t, 
            std::vector<size_t>&, RandomEngine&
        );

    template<class RandomEngine>
        void sampleSubsetWithoutReplacement(const size_t, const size_t, 
            std::vector<size_t>&, RandomEngine&,
            std::vector<size_t>&
        );

    template<typename T>
    T read(std::istream& s, T)
    {
        T value;
        s >> value;

        return value;
    }

    unsigned char read(std::istream& s, unsigned char)
    {
        size_t value;
        s >> value;

        return value;
    }

    char read(std::istream& s, char)
    {
        ptrdiff_t value;
        s >> value;

        return value;
    }

    template<typename T>
    void write(std::ostream& s, T value) const
    {
        s << value;
    }

    void write(std::ostream& s, unsigned char value) const
    {
        s << static_cast<size_t>(label_);
    }

    void write(std::ostream& s, char value) const
    {
        s << static_cast<ptrdiff_t>(label_);
    }


    size_t featureIndex_;
    Feature threshold_;
    size_t childNodeIndices_[2]; // 0 means <, 1 means >=
    Label label_; 
    bool isLeaf_;
};

/// A decision tree.
template<class FEATURE = double, class LABEL = unsigned char>
class DecisionTree {
    
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef DecisionNode<Feature, Label> DecisionNodeType;

    #define LEFT_NODE 0
    #define RIGHT_NODE 1
    using task_t = std::tuple<int, size_t, size_t, size_t, size_t>;
    using in_t = std::tuple<DecisionNodeType, int, size_t, size_t, size_t, size_t, size_t>;


    DecisionTree();
    void learn(const andres::View<Feature>&, const andres::View<Label>&,
            std::vector<size_t>&, const int = 0);
    void learnWithFarm(const andres::View<Feature>&, const andres::View<Label>&,
            std::vector<size_t>&, const int = 0, const int numWorkers = 1);
    void deserialize(std::istream&);

    size_t size() const; // number of decision nodes
    void predict(const andres::View<Feature>&, std::vector<Label>&) const;
    const DecisionNodeType& decisionNode(const size_t) const;
    void serialize(std::ostream&) const;

private:    
    struct TreeConstructionQueueEntry {
        TreeConstructionQueueEntry(
            const size_t nodeIndex = 0, 
            const size_t sampleIndexBegin = 0,
            const size_t sampleIndexEnd = 0,
            const size_t thresholdIndex = 0
        )
        :   nodeIndex_(nodeIndex),
            sampleIndexBegin_(sampleIndexBegin),
            sampleIndexEnd_(sampleIndexEnd),
            thresholdIndex_(thresholdIndex)
        {}

        size_t nodeIndex_;
        size_t sampleIndexBegin_;
        size_t sampleIndexEnd_;
        size_t thresholdIndex_;
    };
    
    struct SourceSink: ff::ff_monode_t<in_t,task_t> {
        

        SourceSink(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            std::vector<size_t>& sampleIndices,
            std::vector<DecisionNodeType>& decisionNodes,
            const int& randomSeed
        )
        :   features_(features),
            labels_(labels),
            sampleIndices_(sampleIndices),
            randomSeed_(randomSeed),
            decisionNodes_(decisionNodes)
        {
            // std::cout << "Starting decision tree construction." << std::endl;
            assert(decisionNodes_.empty());

            {
                decisionNodes_.emplace_back();
                auto thresholdIndex = decisionNodes_.back().learn(
                    features_, labels_, sampleIndices_, 0, sampleIndices_.size(), randomSeed_
                );
                if(!decisionNodes_.back().isLeaf()) {
                    queue_.emplace(0, 0, sampleIndices_.size(), thresholdIndex);
                }
            }
        }

        task_t* svc(in_t* values)  {

            if(values != nullptr) {
                tasksInFlight_.fetch_sub(1);
                auto& [newNode, nodeType, nodeIndexParent, nodeIndexChild, sampleIndexBegin, sampleIndexEnd, thresholdIndex] = *values;
                decisionNodes_[nodeIndexChild] = std::move(newNode);
                if(nodeType == LEFT_NODE) {
                    decisionNodes_[nodeIndexParent].childNodeIndex(LEFT_NODE) = nodeIndexChild;
                } else if(nodeType == RIGHT_NODE) {
                    decisionNodes_[nodeIndexParent].childNodeIndex(RIGHT_NODE) = nodeIndexChild;
                } else {
                    delete values;
                    throw std::runtime_error("Invalid node type in decision tree source sink.");
                }

                if(!decisionNodes_[nodeIndexChild].isLeaf()) {
                    queue_.emplace(nodeIndexChild, sampleIndexBegin, sampleIndexEnd, thresholdIndex);
                }
                delete values;
            }

            if(queue_.empty()) {
                if(tasksInFlight_ == 0){
                    this->broadcast_task(this->EOS);
                    // std::cout << "Decision tree construction finished." << std::endl;
                }
                return this->GO_ON;
            }
            auto entry = queue_.front();
            queue_.pop();

            auto nodeIndexNewLeft = decisionNodes_.size();
            decisionNodes_.emplace_back();
            
            auto nodeIndexNewRight = decisionNodes_.size();
            decisionNodes_.emplace_back();

            this->ff_send_out(
                new task_t(LEFT_NODE, entry.nodeIndex_, nodeIndexNewLeft, entry.sampleIndexBegin_, entry.thresholdIndex_)
            );
            tasksInFlight_.fetch_add(1);
            this->ff_send_out(
                new task_t(RIGHT_NODE, entry.nodeIndex_, nodeIndexNewRight, entry.thresholdIndex_, entry.sampleIndexEnd_)
            );
            tasksInFlight_.fetch_add(1);
            return this->GO_ON;
        }

        void svc_end()  {
            if(decisionNodes_.empty()) {
                throw std::runtime_error("No decision nodes were learned.");
            }
        }

        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        std::vector<size_t>& sampleIndices_;
        std::vector<DecisionNodeType>& decisionNodes_;
        const int randomSeed_;
        std::queue<TreeConstructionQueueEntry> queue_;
        std::atomic<size_t> tasksInFlight_{0};
    };

     struct Worker: ff::ff_node_t<task_t, in_t> {
        

        Worker(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            std::vector<size_t>& sampleIndices,
            const int& randomSeed
        )
        :   features_(features),
            labels_(labels),
            sampleIndices_(sampleIndices),
            randomSeed_(randomSeed)
        {}


        in_t* svc(task_t* task) {
            auto [nodeType, nodeIndexParent, nodeIndexChild, sampleIndexBegin, sampleIndexEnd] = *task;
            delete task;

            DecisionNodeType newNode;
            size_t thresholdIndex = newNode.learn(
                features_, labels_, sampleIndices_,
                sampleIndexBegin, sampleIndexEnd, randomSeed_
            );

            in_t* result = new in_t(
                std::move(newNode), nodeType, nodeIndexParent, nodeIndexChild, 
                sampleIndexBegin, sampleIndexEnd, thresholdIndex
            );
            this->ff_send_out(result);
            return this->GO_ON;
        }

        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        std::vector<size_t>& sampleIndices_;
        const int randomSeed_;
        
    };

    

    std::vector<DecisionNodeType> decisionNodes_;
};

/// A bag of decision trees.
template<
    class FEATURE = double, 
    class LABEL = unsigned char, 
    class PROBABILITY = double
>
class DecisionForest {
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef PROBABILITY Probability;
    typedef DecisionTree<Feature, Label> DecisionTreeType;

    DecisionForest();
    void clear();    
    void learn(const andres::View<Feature>&, const andres::View<Label>&,
            const size_t, const int = 0);
    void deserialize(std::istream&);

    size_t size() const;
    const DecisionTreeType& decisionTree(const size_t) const;
    void predict(const andres::View<Feature>&, andres::Marray<Probability>&) const;
    void serialize(std::ostream&) const;

private:
    template<class RandomEngine>
        void sampleBootstrap(const size_t, std::vector<size_t>&, RandomEngine&);

    std::vector<DecisionTreeType> decisionTrees_;
};

// implementation of DecisionNode

/// Constructs a decision node.
/// 
template<class FEATURE, class LABEL>
inline
DecisionNode<FEATURE, LABEL>::DecisionNode()
:   featureIndex_(), 
    threshold_(), 
    label_(), 
    isLeaf_(false)
{
    childNodeIndices_[0] = 0;
    childNodeIndices_[1] = 0;
}

/// Returns true if the node is a leaf node.
/// 
template<class FEATURE, class LABEL>
inline bool 
DecisionNode<FEATURE, LABEL>::isLeaf() const {
    return isLeaf_;
}

/// Returns true if the node is a leaf node.
/// 
template<class FEATURE, class LABEL>
inline bool& 
DecisionNode<FEATURE, LABEL>::isLeaf() {
    return isLeaf_;
}

/// Returns, for a non-leaf node, the index of a feature wrt which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline size_t 
DecisionNode<FEATURE, LABEL>::featureIndex() const {
    assert(!isLeaf());
    return featureIndex_;
}

/// Returns, for a non-leaf node, the index of a feature wrt which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline size_t& 
DecisionNode<FEATURE, LABEL>::featureIndex() {
    assert(!isLeaf());
    return featureIndex_;
}

/// Returns, for a non-leaf node, a threshold by which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Feature 
DecisionNode<FEATURE, LABEL>::threshold() const {
    assert(!isLeaf());
    return threshold_;
}

/// Returns, for a non-leaf node, a threshold by which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Feature& 
DecisionNode<FEATURE, LABEL>::threshold() {
    assert(!isLeaf());
    return threshold_;
}

/// Returns, for a non-leaf node, the index of one of its two child nodes.
///
/// \param j number of the child node (either 0 or 1).
/// 
template<class FEATURE, class LABEL>
inline size_t 
DecisionNode<FEATURE, LABEL>::childNodeIndex(
    const size_t j
) const {
    assert(!isLeaf());
    assert(j < 2);
    return childNodeIndices_[j];
}

/// Returns, for a non-leaf node, the index of one of its two child nodes.
///
/// \param j number of the child node (either 0 or 1).
/// 
template<class FEATURE, class LABEL>
inline size_t& 
DecisionNode<FEATURE, LABEL>::childNodeIndex(
    const size_t j
) {
    assert(!isLeaf());
    assert(j < 2);
    return childNodeIndices_[j];
}

/// Returns, for a leaf node, its label.
///
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Label
DecisionNode<FEATURE, LABEL>::label() const {
    assert(isLeaf());
    return label_;
}

/// Returns, for a leaf node, its label.
///
template<class FEATURE, class LABEL>
inline typename DecisionNode<FEATURE, LABEL>::Label& 
DecisionNode<FEATURE, LABEL>::label() {
    assert(isLeaf());
    return label_;
}

/// Learns a decision node from labeled samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
/// \param sampleIndexBegin Index of the first element of sampleIndices to be considered.
/// \param sampleIndexEnd Index one greater than that of the last element of sampleIndices to be considered.
/// \param randomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
///
template<class FEATURE, class LABEL>
size_t
DecisionNode<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices, // input, will be sorted
    const size_t sampleIndexBegin,
    const size_t sampleIndexEnd,
    const int randomSeed
) {
    /*
    std::cout << "learning new node from sample indices";
    for(size_t j = sampleIndexBegin; j < sampleIndexEnd; ++j) {
        std::cout << " " << sampleIndices[j];
    }
    std::cout << std::endl;
    */

    assert(sampleIndices.size() != 0);
    assert(sampleIndexEnd <= sampleIndices.size());
    assert(sampleIndexBegin < sampleIndexEnd);
    assert(features.shape(0) != 0);
    assert(features.shape(0) == labels.size());

    // handle the case in which there is only one unique label (pure node)
    {
        bool isLabelUnique = true;
        const size_t firstLabelSampleIndex = sampleIndices[sampleIndexBegin];
        // Ensure firstLabelSampleIndex is valid before dereferencing labels
        if (firstLabelSampleIndex >= labels.size()) {
            throw std::out_of_range("Sample index out of bounds for labels.");
        }
        const Label firstLabel = labels(firstLabelSampleIndex);
        for(size_t j = sampleIndexBegin + 1; j < sampleIndexEnd; ++j) {
            const size_t currentSampleIndex = sampleIndices[j];
            if (currentSampleIndex >= labels.size()) {
                 throw std::out_of_range("Sample index out of bounds for labels during purity check.");
            }
            if(labels(currentSampleIndex) != firstLabel) { 
                isLabelUnique = false;
                break;
            }
        }
        if(isLabelUnique) {
            isLeaf_ = true;
            label_ = firstLabel;
            // std::cout << "sample is pure." << std::endl;
            return 0;
        }
    }
    
    const size_t numberOfFeatures = features.shape(1);
    if (numberOfFeatures == 0) { // Handle case with no features
        isLeaf_ = true;
        // Set label to majority label, or first label if only one sample type
        std::map<Label, size_t> localLabelCounts;
        for (size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
            localLabelCounts[labels(sampleIndices[k])]++;
        }
        Label majorityLabelValue = labels(sampleIndices[sampleIndexBegin]);
        size_t maxCount = 0;
        for (const auto& pair : localLabelCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                majorityLabelValue = pair.first;
            }
        }
        label_ = majorityLabelValue;
        return 0;
    }

    Label max_label = 0;
    if (sampleIndexBegin < sampleIndexEnd) {
        max_label = labels(sampleIndices[sampleIndexBegin]);
        for (size_t k = sampleIndexBegin + 1; k < sampleIndexEnd; ++k) {
            if (labels(sampleIndices[k]) > max_label) {
                max_label = labels(sampleIndices[k]);
            }
        }
    }
    const size_t numberOfClassesInSubset = static_cast<size_t>(max_label) + 1;
    

    const size_t numberOfFeaturesToBeAssessed = 
        static_cast<size_t>(
            std::ceil(std::sqrt(
                static_cast<double>(numberOfFeatures)
            ))
        );

    std::vector<size_t> featureIndicesBuffer(numberOfFeaturesToBeAssessed);
    std::vector<SortItem> sortBuffer;
    
    std::vector<size_t> randomSampleBuffer; 
    auto randomEngine = 
        (randomSeed == 0) ? std::mt19937(std::random_device{}()) 
                          : std::mt19937(randomSeed);
    sampleSubsetWithoutReplacement(
        numberOfFeatures, 
        numberOfFeaturesToBeAssessed, 
        featureIndicesBuffer,
        randomEngine,
        randomSampleBuffer
    );

    std::vector<size_t> numbersOfLabelsForSplit[2]; 
    numbersOfLabelsForSplit[0].reserve(10); 
    numbersOfLabelsForSplit[1].reserve(10); 
    double optimalSumOfGiniCoefficients = std::numeric_limits<double>::infinity();
    size_t currentOptimalFeatureIndex = 0; // Initialize
    Feature currentOptimalThreshold = Feature(); // Initialize
    size_t currentOptimalThresholdIndex = sampleIndexBegin; // Initialize
    

    for(size_t j = 0; j < numberOfFeaturesToBeAssessed; ++j) {
        const size_t fi = featureIndicesBuffer[j];

        // To improve cache performance, create a temporary vector of structs
        // to hold feature values and indices, sort it, then update sampleIndices.
        // This makes the sort operation much faster by avoiding strided memory access.
        sortBuffer.clear();
        const size_t numSamplesInNode = sampleIndexEnd - sampleIndexBegin;
        sortBuffer.reserve(numSamplesInNode);
        for (size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
            sortBuffer.emplace_back(features(sampleIndices[k], fi), sampleIndices[k]);
        }
        std::sort(sortBuffer.begin(), sortBuffer.end());

        // Update sampleIndices to reflect the new sorted order.
        for (size_t k = 0; k < numSamplesInNode; ++k) {
            sampleIndices[sampleIndexBegin + k] = sortBuffer[k].sampleIndex;
        }

        // std::sort(
        //     sampleIndices.begin() + sampleIndexBegin, 
        //     sampleIndices.begin() + sampleIndexEnd, 
        //     ComparisonByFeature(features, fi)
        // );
        #ifndef NDEBUG
        for(size_t k = sampleIndexBegin; k + 1 < sampleIndexEnd; ++k) {
            assert(
                features(sampleIndices[k], fi) <= features(sampleIndices[k + 1], fi)
            );
        }
        #endif

        size_t numbersOfElements[] = {0, sampleIndexEnd - sampleIndexBegin};
        
        for(size_t s = 0; s < 2; ++s) { // Clear/resize for current feature
            numbersOfLabelsForSplit[s].assign(numberOfClassesInSubset, 0);
        }

        for(size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
            const Label label = labels(sampleIndices[k]);
            // if(label >= numbersOfLabelsForSplit[1].size()) { // Should be handled by pre-sizing
            //      numbersOfLabelsForSplit[0].resize(label + 1, 0);
            //      numbersOfLabelsForSplit[1].resize(label + 1, 0);
            // }
            ++numbersOfLabelsForSplit[1][label];
        }

        size_t thresholdIndexLoopVar = sampleIndexBegin;
        for(;;) { 
            const size_t thresholdIndexOld = thresholdIndexLoopVar;

            while(thresholdIndexLoopVar + 1 < sampleIndexEnd
            && features(sampleIndices[thresholdIndexLoopVar], fi) 
            == features(sampleIndices[thresholdIndexLoopVar + 1], fi)) {                
                const Label label = labels(sampleIndices[thresholdIndexLoopVar]);
                // if (label >= numbersOfLabelsForSplit[0].size() || label >= numbersOfLabelsForSplit[1].size()) {
                //     size_t newSize = label + 1;
                //     if (numbersOfLabelsForSplit[0].size() < newSize) numbersOfLabelsForSplit[0].resize(newSize, 0);
                //     if (numbersOfLabelsForSplit[1].size() < newSize) numbersOfLabelsForSplit[1].resize(newSize, 0);
                // }
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
                ++thresholdIndexLoopVar;
            }

            {
                const Label label = labels(sampleIndices[thresholdIndexLoopVar]);
                //  if (label >= numbersOfLabelsForSplit[0].size() || label >= numbersOfLabelsForSplit[1].size()) {
                //     size_t newSize = label + 1;
                //     if (numbersOfLabelsForSplit[0].size() < newSize) numbersOfLabelsForSplit[0].resize(newSize, 0);
                //     if (numbersOfLabelsForSplit[1].size() < newSize) numbersOfLabelsForSplit[1].resize(newSize, 0);
                // }
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
            }
            ++thresholdIndexLoopVar; 
            if(thresholdIndexLoopVar == sampleIndexEnd) {
                break;
            }

            assert(numbersOfLabelsForSplit[0].size() == numbersOfLabelsForSplit[1].size());
            size_t numbersOfDistinctPairs[] = {0, 0};
            for(size_t s = 0; s < 2; ++s) 
            for(size_t k_label = 0; k_label < numbersOfLabelsForSplit[s].size(); ++k_label)
            for(size_t m_label = k_label + 1; m_label < numbersOfLabelsForSplit[s].size(); ++m_label) {
                numbersOfDistinctPairs[s] += 
                    numbersOfLabelsForSplit[s][k_label] * numbersOfLabelsForSplit[s][m_label];
            }

            double giniCoefficients[2];
            for(size_t s = 0; s < 2; ++s) {
                if(numbersOfElements[s] < 2) {
                    giniCoefficients[s] = 0;
                }
                else {
                    giniCoefficients[s] = 
                        static_cast<double>(numbersOfDistinctPairs[s])
                        / (static_cast<double>(numbersOfElements[s]) * (numbersOfElements[s] - 1));
                }
            }

            double sumOfginiCoefficients = giniCoefficients[0] + giniCoefficients[1];
            if(sumOfginiCoefficients < optimalSumOfGiniCoefficients) {
                optimalSumOfGiniCoefficients = sumOfginiCoefficients;
                currentOptimalFeatureIndex = fi;
                currentOptimalThreshold = features(sampleIndices[thresholdIndexLoopVar], fi); 
                currentOptimalThresholdIndex = thresholdIndexLoopVar;
            }
        }
        for(size_t s = 0; s < 2; ++s) {
            std::fill(numbersOfLabelsForSplit[s].begin(), numbersOfLabelsForSplit[s].end(), 0);
        }
    }

    if (optimalSumOfGiniCoefficients == std::numeric_limits<double>::infinity()) {
        // No effective split found. Make this node a leaf.
        isLeaf_ = true;
        std::map<Label, size_t> localLabelCounts;
        for (size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
            localLabelCounts[labels(sampleIndices[k])]++;
        }
        
        Label majorityLabelValue = labels(sampleIndices[sampleIndexBegin]); // Default
        size_t maxCount = 0;
        // Ensure there's at least one sample before iterating map (already asserted by sampleIndexBegin < sampleIndexEnd)
        for (const auto& pair : localLabelCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                majorityLabelValue = pair.first;
            }
        }
        label_ = majorityLabelValue;
        return 0; // Leaf nodes return 0
    }

    // A split was found
    this->featureIndex_ = currentOptimalFeatureIndex;
    this->threshold_ = currentOptimalThreshold;

    sortBuffer.clear();
    for (size_t k = sampleIndexBegin; k < sampleIndexEnd; ++k) {
        sortBuffer.emplace_back(features(sampleIndices[k], this->featureIndex_), sampleIndices[k]);
    }

    // 2. Sort: Sort the contiguous buffer. This is very fast due to data locality.
    std::sort(sortBuffer.begin(), sortBuffer.end());

    // 3. Scatter: Write the sorted indices back into the main sampleIndices vector.
    const size_t numSamplesInNode = sampleIndexEnd - sampleIndexBegin;
    for (size_t k = 0; k < numSamplesInNode; ++k) {
        sampleIndices[sampleIndexBegin + k] = sortBuffer[k].sampleIndex;
    }
    
    return currentOptimalThresholdIndex;
}

template<class FEATURE, class LABEL>
template<class RandomEngine>
inline void 
DecisionNode<FEATURE, LABEL>::sampleSubsetWithoutReplacement(
    const size_t size,
    const size_t subsetSize,
    std::vector<size_t>& indices,
    RandomEngine& randomEngine
) {
    std::vector<size_t> buffer;
    sampleSubsetWithoutReplacement(size, subsetSize, indices, randomEngine, buffer);
}

template<class FEATURE, class LABEL>
template<class RandomEngine>
inline void 
DecisionNode<FEATURE, LABEL>::sampleSubsetWithoutReplacement(
    const size_t size,
    const size_t subsetSize,
    std::vector<size_t>& indices, // output
    RandomEngine& randomEngine,
    std::vector<size_t>& candidateIndices // buffer
) {
    assert(subsetSize <= size);
    indices.resize(subsetSize);

    // start with indices {0, ..., size - 1} as candidates
    candidateIndices.resize(size);
    for(size_t j = 0; j < size; ++j) {
        candidateIndices[j] = j;
    }

    // draw "subsetSize" indices without replacement
    for(size_t j = 0; j < subsetSize; ++j) {
        std::uniform_int_distribution<size_t> distribution(0, size - j - 1);
        const size_t index = distribution(randomEngine);
        indices[j] = candidateIndices[index];
        candidateIndices[index] = candidateIndices[size - j - 1];
        #ifndef NDEBUG
        for(size_t k = 0; k < j; ++k) {
            assert(indices[k] != indices[j]);
        }
        #endif
    }
}

/// Serialization.
///
template<class FEATURE, class LABEL>
inline void
DecisionNode<FEATURE, LABEL>::serialize(std::ostream& s) const
{
    s << " " << featureIndex_;
    s << " ";
    write(s, threshold_);
    s << " " << childNodeIndices_[0];
    s << " " << childNodeIndices_[1];
    s << " ";
    write(s, label_);
    s << " " << isLeaf_;
}

/// De-serialization.
///
template<class FEATURE, class LABEL>
inline void
DecisionNode<FEATURE, LABEL>::deserialize(std::istream& s)
{
    s >> featureIndex_;
    threshold_ = read(s, FEATURE());
    s >> childNodeIndices_[0];
    s >> childNodeIndices_[1];
    label_ = read(s, LABEL());
    s >> isLeaf_;
}

// implementation of DecisionTree

/// Constructs a decision tree.
///
template<class FEATURE, class LABEL>
inline
DecisionTree<FEATURE, LABEL>::DecisionTree()
:   decisionNodes_()
{}


/// Learns a decision tree as described by Leo Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
/// \param randomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
///
template<class FEATURE, class LABEL>
void 
DecisionTree<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices, // input, will be sorted
    const int randomSeed
) {
    assert(decisionNodes_.size() == 0);
   
    std::queue<TreeConstructionQueueEntry> queue;
    // learn root node
    {
        decisionNodes_.push_back(DecisionNodeType());
        size_t thresholdIndex = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices, 
            0, sampleIndices.size(),
            randomSeed
        );        
        if(!decisionNodes_[0].isLeaf()) { // if root node is not pure
            queue.push(
                TreeConstructionQueueEntry(
                    0, // node index
                    0, sampleIndices.size(), // range of samples
                    thresholdIndex 
                )
            );
        }
    }
    while(!queue.empty()) {
        const size_t nodeIndex = queue.front().nodeIndex_;
        const size_t sampleIndexBegin = queue.front().sampleIndexBegin_;
        const size_t sampleIndexEnd = queue.front().sampleIndexEnd_;
        const size_t thresholdIndex = queue.front().thresholdIndex_;
        queue.pop();

        size_t nodeIndexNew;
        size_t thresholdIndexNew;

        // learn left child node and put on queue
        nodeIndexNew = decisionNodes_.size();
        decisionNodes_.push_back(DecisionNodeType());
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices,
            sampleIndexBegin, thresholdIndex,
            randomSeed
        );
        #ifndef NDEBUG
        if(decisionNodes_[nodeIndexNew].isLeaf()) {
            assert(thresholdIndexNew == 0);
        }
        else {
            assert(
                thresholdIndexNew >= sampleIndexBegin
                && thresholdIndexNew < thresholdIndex
            );
        }
        #endif
        decisionNodes_[nodeIndex].childNodeIndex(0) = nodeIndexNew;
        if(!decisionNodes_[nodeIndexNew].isLeaf()) { // if not pure
            queue.push(
                TreeConstructionQueueEntry(
                    nodeIndexNew,
                    sampleIndexBegin, thresholdIndex,
                    thresholdIndexNew
                )
            );
        }

        // learn right child node and put on queue
        nodeIndexNew = decisionNodes_.size();
        decisionNodes_.push_back(DecisionNodeType());
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndices,
            thresholdIndex, sampleIndexEnd,
            randomSeed
        );
        #ifndef NDEBUG
        if(decisionNodes_[nodeIndexNew].isLeaf()) {
            assert(thresholdIndexNew == 0);
        }
        else {
            assert(
                thresholdIndexNew >= thresholdIndex
                && thresholdIndexNew < sampleIndexEnd
            );
        }
        #endif
        decisionNodes_[nodeIndex].childNodeIndex(1) = nodeIndexNew;
        if(!decisionNodes_[nodeIndexNew].isLeaf()) { // if not pure
            queue.push(
                TreeConstructionQueueEntry(
                    nodeIndexNew,
                    thresholdIndex, sampleIndexEnd,
                    thresholdIndexNew
                )
            );
        }
    }
}

/// Learns a decision tree using FastFlow Farm for parallel node processing.
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param sampleIndices A sequence of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
/// \param randomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
/// \param numWorkers Number of workers for the farm (should be less than total available workers)
///
template<class FEATURE, class LABEL>
void 
DecisionTree<FEATURE, LABEL>::learnWithFarm(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    std::vector<size_t>& sampleIndices, // input, will be sorted
    const int randomSeed,
    const int numWorkers
) {
    assert(decisionNodes_.size() == 0);
    
   SourceSink sourceSink(
        features, labels, sampleIndices, decisionNodes_, randomSeed
    );

    // Initialize the farm with the source sink and workers
    ff::ff_Farm<task_t, in_t> farm(
        [&](){
            std::vector<std::unique_ptr<ff::ff_node>> workers;
            for(int i = 0; i < numWorkers; ++i) {
                workers.push_back(
                    std::make_unique<Worker>(features, labels, sampleIndices, randomSeed)
                );
            }
            return std::move(workers);
        } (),
        sourceSink
    );
    
    farm.remove_collector(); 
    farm.wrap_around();
    farm.set_scheduling_ondemand(); 

    // Run the farm
    if(farm.run_and_wait_end() < 0) {
        throw std::runtime_error("Error during decision tree learning with Farm.");
    }

    // Check if any decision nodes were learned
    if(decisionNodes_.empty()) {
        throw std::runtime_error("No decision nodes were learned.");
    }
}

/// Returns the number of decision nodes.
///
template<class FEATURE, class LABEL>
inline size_t 
DecisionTree<FEATURE, LABEL>::size() const {
    return decisionNodes_.size();
}

/// Returns a decision node.
///
template<class FEATURE, class LABEL>
inline const typename DecisionTree<FEATURE, LABEL>::DecisionNodeType& 
DecisionTree<FEATURE, LABEL>::decisionNode(
    const size_t decisionNodeIndex
) const {
    return decisionNodes_[decisionNodeIndex];
}

/// Predicts the labels of feature vectors.
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample. (output)
///
template<class FEATURE, class LABEL>
inline void 
DecisionTree<FEATURE, LABEL>::predict(
    const andres::View<Feature>& features,
    std::vector<Label>& labels
) const  {
    const size_t numberOfSamples = features.shape(0);
    const size_t numberOfFeatures = features.shape(1);
    labels.resize(numberOfSamples);
    for(size_t j = 0; j < numberOfSamples; ++j) {
        size_t nodeIndex = 0;
        for(;;) {
            const DecisionNodeType& decisionNode = decisionNodes_[nodeIndex];
            if(decisionNode.isLeaf()) {
                labels[j] = decisionNode.label();
                break;
            }
            else {
                const size_t fi = decisionNode.featureIndex();
                const Feature threshold = decisionNode.threshold();
                assert(fi < numberOfFeatures);
                if(features(j, fi) < threshold) {
                    nodeIndex = decisionNode.childNodeIndex(0);
                }
                else {
                    nodeIndex = decisionNode.childNodeIndex(1);
                }
                assert(nodeIndex != 0); // assert that tree is not incomplete
            }
        }
    }
}

/// Serialization.
///
template<class FEATURE, class LABEL>
inline void
DecisionTree<FEATURE, LABEL>::serialize(std::ostream& s) const {
    s << " " << decisionNodes_.size();
    for(size_t j = 0; j < decisionNodes_.size(); ++j) {
        decisionNodes_[j].serialize(s);
    }
}

/// De-serialization.
///
template<class FEATURE, class LABEL>
inline void
DecisionTree<FEATURE, LABEL>::deserialize(std::istream& s) {
    size_t numberOfDecisionNodes = 0;
    s >> numberOfDecisionNodes;
    decisionNodes_.resize(numberOfDecisionNodes);
    for(size_t j = 0; j < numberOfDecisionNodes; ++j) {
        decisionNodes_[j].deserialize(s);
    }
}

// implementation of DecisionForest

/// Constructs a decision forest.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline
DecisionForest<FEATURE, LABEL, PROBABILITY>::DecisionForest()
:   decisionTrees_()
{}

/// Clears a decision forest.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void
DecisionForest<FEATURE, LABEL, PROBABILITY>::clear() {
    decisionTrees_.clear();
}

/// Returns the number of decision trees.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline size_t
DecisionForest<FEATURE, LABEL, PROBABILITY>::size() const {
    return decisionTrees_.size();
}


/// Learns a decision forest from labeled samples using nested parallelism.
/// Uses ParallelFor for tree-level parallelism and Farm for node-level parallelism.
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labels A vector of labels, one for each sample.
/// \param numberOfDecisionTrees Number of decision trees to be learned.
/// \param randomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    const size_t numberOfDecisionTrees,
    const int randomSeed
) {
    if(features.dimension() != 2) {
        throw std::runtime_error("features.dimension() != 2");
    }
    if(labels.dimension() != 1) {
        throw std::runtime_error("labels.dimension() != 1");
    }
    if(features.shape(0) != labels.size()) {
        throw std::runtime_error("the number of samples does not match the size of the label vector.");
    }
    const size_t numberOfSamples = features.shape(0);
    const size_t numberOfFeatures = features.shape(1);

    // Decide on nested parallelism strategy based on problem size
    bool useNestedParallelism =
                                     (numberOfSamples > 20000) && 
                                     (numberOfFeatures > 10) &&
                                     (numberOfDecisionTrees > 1);

    // Calculate worker distribution for nested parallelism
   
    const int totalCores = ff_numCores();
    const int forestWorkers = FOR_NUM_WORKERS;
    
    int workersPerTree = 2;
    if (useNestedParallelism) {
        if (forestWorkers > 0) {
            const int coresPerFarm = totalCores / forestWorkers;
            const int potentialWorkers = std::min(FARM_NUM_WORKERS, coresPerFarm);
            // Only enable the farm if it can run with at least 2 workers.
            if (potentialWorkers >= 2) {
                workersPerTree = potentialWorkers;
            } else {
                useNestedParallelism = false; // Disable nested parallelism if not enough workers
            }
        }
    }

    
              
    clear();
    decisionTrees_.resize(numberOfDecisionTrees);
    ff::ParallelFor pf(forestWorkers, FOR_SPINWAIT, FOR_SPINBARRIER);
    pf.parallel_for(
        0, static_cast<ptrdiff_t>(decisionTrees_.size()), 1, FOR_CHUNK_SIZE,
        [&](ptrdiff_t treeIndex) {
            std::vector<size_t> sampleIndices(numberOfSamples);
            
            int tree_specific_seed = (randomSeed == 0) ? 0 : (randomSeed + treeIndex);
            std::mt19937 randomEngine;
            if (tree_specific_seed == 0) {
                std::seed_seq seq{(unsigned int)std::random_device{}(), (unsigned int)treeIndex};
                randomEngine.seed(seq);
            } else {
                randomEngine.seed(tree_specific_seed);
            }
            
            sampleBootstrap(numberOfSamples, sampleIndices, randomEngine);
            // std::cout << "Using " << (useNestedParallelism ? "learning with farm" : "learning without farm") 
            //           << " for tree " << treeIndex << std::endl; 
            // Choose learning strategy based on nested parallelism decision
            if (useNestedParallelism) {
                decisionTrees_[treeIndex].learnWithFarm(
                    features, labels, sampleIndices, tree_specific_seed, workersPerTree
                );
            } else {
                decisionTrees_[treeIndex].learn(
                    features, labels, sampleIndices, tree_specific_seed
                );
            }
        }, forestWorkers
    ); 
    // for(ptrdiff_t treeIndex = 0; treeIndex < static_cast<ptrdiff_t>(decisionTrees_.size()); ++treeIndex) {
    //     std::vector<size_t> sampleIndices(numberOfSamples);
    //     sampleBootstrap(numberOfSamples, sampleIndices, randomEngine?????);
    //     decisionTrees_[treeIndex].learn(features, labels, sampleIndices, randomSeed);
    // }
}

/// Predict the label probabilities of samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labelProbabilities A matrix of probabilities in which every rows corresponds to a sample and every column corresponds to a label.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::predict(
    const andres::View<Feature>& features,
    andres::Marray<Probability>& labelProbabilities 
) const  {
    if(size() == 0) {
        throw std::runtime_error("no decision trees.");
    }
    if(features.dimension() != 2) {
        throw std::runtime_error("features.dimension() != 2");
    }
    if(features.shape(0) != labelProbabilities.shape(0)) {
        throw std::runtime_error("labelProbabilities.shape(0) does not match the number of samples.");
    }

    const size_t numberOfSamples = features.shape(0);
    const size_t numberOfFeatures = features.shape(1);
    std::fill(labelProbabilities.begin(), labelProbabilities.end(), Probability());
    
    for(ptrdiff_t treeIndex = 0; treeIndex < static_cast<ptrdiff_t>(decisionTrees_.size()); ++treeIndex) {
        std::vector<Label> labels(numberOfSamples);
        const DecisionTreeType& decisionTree = decisionTrees_[treeIndex];
        decisionTree.predict(features, labels);
        for(size_t sampleIndex = 0; sampleIndex < numberOfSamples; ++sampleIndex) {
            const Label label = labels[sampleIndex];
            if(label >= labelProbabilities.shape(1)) {
                throw std::runtime_error("labelProbabilities.shape(1) does not match the number of labels.");
            }
           
            ++labelProbabilities(sampleIndex, label);
        }
    }
    
    for(ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(labelProbabilities.size()); ++j) {
        labelProbabilities(j) /= decisionTrees_.size();
    }
}

/// Returns a decision tree.
///
/// \param treeIndex Index of the decisio tree.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline const typename DecisionForest<FEATURE, LABEL, PROBABILITY>::DecisionTreeType& 
DecisionForest<FEATURE, LABEL, PROBABILITY>::decisionTree(
    const size_t treeIndex
) const {
    return decisionTrees_[treeIndex];
}

// draw "size" out of "size", with replacement
template<class FEATURE, class LABEL, class PROBABILITY>
template<class RandomEngine>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::sampleBootstrap(
    const size_t size,
    std::vector<size_t>& indices,
    RandomEngine& randomEngine
) {
    indices.resize(size);    
   
    for(size_t j = 0; j < size; ++j) {
        std::uniform_int_distribution<size_t> distribution(0, size - 1);
        indices[j] = distribution(randomEngine);
        assert(indices[j] < size);
    }
}

/// Serialization.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void
DecisionForest<FEATURE, LABEL, PROBABILITY>::serialize(std::ostream& s) const {
    s << decisionTrees_.size();
    for(size_t j = 0; j < decisionTrees_.size(); ++j) {
        decisionTrees_[j].serialize(s);
    }
}

/// De-serialization.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void
DecisionForest<FEATURE, LABEL, PROBABILITY>::deserialize(std::istream& s) {
    size_t numberOfDecisionTrees;
    s >> numberOfDecisionTrees;
    decisionTrees_.resize(numberOfDecisionTrees);
    for(size_t j = 0; j < numberOfDecisionTrees; ++j) {
        decisionTrees_[j].deserialize(s);
    }
}

} // namespace ml
} // namespace andres

#endif // #ifndef ANDRES_ML_DECISION_FOREST_HXX