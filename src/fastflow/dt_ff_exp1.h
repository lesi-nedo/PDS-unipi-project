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
#ifndef ANDRES_ML_FF_EXP1_DECISION_FOREST_HXX
#define ANDRES_ML_FF_EXP1_DECISION_FOREST_HXX

#include <stdexcept>
#include <random>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <unordered_map>
#include <tuple>
#include <immintrin.h>
#include <thread>
#include <variant>
#include <execution>


#include "ff/ff.hpp"
#include "ff/parallel_for.hpp"
#include "ff/farm.hpp"
#include "ff/map.hpp"

#include "ff_impl_config.h"

#include "marray.h"

#define LEFT_NODE 0
#define RIGHT_NODE 1



/// The public API.
namespace andres {
    
/// Machine Learning.
namespace ff_ml_exp1 {
    template<typename F>
    struct SortItem {
        F featureValue;
        size_t sampleIndex;
        bool operator<(const SortItem& other) const {
                return featureValue < other.featureValue;
        }
    };
    
template<class FEATURE, class LABEL, class PROBABILITY>
class DecisionForest;

/// A node in a decision tree.
template<class FEATURE, class LABEL>
class DecisionNode {
    public:
        typedef FEATURE Feature;
        typedef LABEL Label;
 
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

    template<class RandomEngine>
        static void sampleSubsetWithoutReplacement(const size_t, const size_t, 
            std::vector<size_t>&, RandomEngine&
        );

    template<class RandomEngine>
        static void sampleSubsetWithoutReplacement(const size_t, const size_t, 
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

    template<std::ranges::range Inds>
    inline std::tuple<double, size_t, FEATURE, size_t> helperSequential(
        Inds&, const size_t&, const size_t&, const size_t&,
        std::vector<SortItem<Feature>>&, std::vector<size_t>&,
        const andres::View<Feature>&, const andres::View<Label>&
    );
    template<std::ranges::range Inds>
    std::tuple<double, size_t, FEATURE, size_t> helperParallelMap(
        const size_t&, const size_t&, const size_t&, const size_t&,
        std::vector<size_t>&, Inds&,
        const andres::View<Feature>&, const andres::View<Label>&
    );


    size_t featureIndex_;
    Feature threshold_;
    size_t childNodeIndices_[2]; // 0 means <, 1 means >=
    Label label_; 
    bool isLeaf_;

public:

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
    

    template<std::ranges::range Inds>
    size_t learn(
        const andres::View<Feature>&,
        const andres::View<Label>&,
        Inds&,
        const size_t&,
        const int&
    );
    void serialize(std::ostream&) const;

    struct BestSplit {
        double gini = std::numeric_limits<double>::infinity();
        double balance = std::numeric_limits<double>::infinity(); 
        size_t featureIndex = 0;
        Feature threshold = Feature();
        size_t relativeThresholdIndex = 0;

        bool operator<(const BestSplit& other) const {
            return gini < other.gini;
        }
    };
        
};

/// A decision tree.
template<class FEATURE = double, class LABEL = unsigned char>
class DecisionTree {
    
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef DecisionNode<Feature, Label> DecisionNodeType;

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

private:    
    
    struct SourceSink: ff::ff_monode_t<in_t,task_t> {
        

        SourceSink(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            std::vector<size_t> sampleIndices,
            std::vector<DecisionNodeType>& decisionNodes,
            const int& randomSeed
        )
        :   features_(features),
            labels_(labels),
            sampleIndices_(sampleIndices),
            randomSeed_(randomSeed),
            decisionNodes_(decisionNodes),
            sortItems_(features.shape(0))
        {
            assert(decisionNodes_.empty());

            {
                decisionNodes_.emplace_back();
                auto sampleIndicesView = std::ranges::subrange(
                    sampleIndices_.begin(), 
                    sampleIndices_.end()
                );
                auto thresholdIndex = decisionNodes_.back().learn(
                    features_, labels_,sampleIndicesView, 0, randomSeed_
                );
                if(!decisionNodes_.back().isLeaf()) {
                    queue_.emplace(0,0, sampleIndices_.size(), thresholdIndex);
                }
            }
        }

        task_t* svc(in_t* values)  {

            if(values != nullptr) {
                tasksInFlight_ -=1;
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
                const size_t numItems = sampleIndexEnd - sampleIndexBegin;
                for(size_t ind = 0; ind < numItems; ++ind) {
                    sortItems_[ind] = std::move(SortItem<Feature>(
                            features_(sampleIndexBegin + ind, newNode.featureIndex()), sampleIndexBegin + ind
                    ));
                }
                std::sort(sortItems_.begin(), sortItems_.begin() + numItems);
                for(size_t ind = 0; ind < numItems; ++ind) {
                    #if defined(DEBUG)
                    assert(sampleIndices_.size() > ind + sampleIndexBegin);
                    #endif
                    sampleIndices_[ind + sampleIndexBegin] = sortItems_[ind].sampleIndex;
                    
                }
                
                delete values;
            }

            if(queue_.empty()) {
                if(tasksInFlight_ <= 0){
                    this->broadcast_task(this->EOS);
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
                new task_t(
                    LEFT_NODE, entry.nodeIndex_, nodeIndexNewLeft, entry.sampleIndexBegin_, entry.thresholdIndex_
                )
            );
            this->ff_send_out(
                new task_t(
                    RIGHT_NODE, entry.nodeIndex_, nodeIndexNewRight, entry.thresholdIndex_, entry.sampleIndexEnd_
                )
            );
            tasksInFlight_ += 2;
            return this->GO_ON;
        }

        void svc_end()  {
            if(decisionNodes_.empty()) {
                throw std::runtime_error("No decision nodes were learned.");
            }
        }

        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        std::vector<size_t> sampleIndices_;
        std::vector<DecisionNodeType>& decisionNodes_;
        const int randomSeed_;
        std::queue<TreeConstructionQueueEntry> queue_;
        int tasksInFlight_{0};
        std::vector<SortItem<Feature>> sortItems_;
    };

     struct Worker: ff::ff_node_t<task_t, in_t> {
        

        Worker(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            const std::vector<size_t>& sampleIndices,
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

            // Create a view of the slice for the worker
            std::vector<size_t> sampleIndicesCopy = std::vector<size_t>(
                sampleIndices_.begin() + sampleIndexBegin,
                sampleIndices_.begin() + sampleIndexEnd
            );
            size_t thresholdIndex = newNode.learn(
                features_, labels_, sampleIndicesCopy, sampleIndexBegin, randomSeed_
            );

            return new in_t(
                std::move(newNode), nodeType, nodeIndexParent, nodeIndexChild,
                sampleIndexBegin, sampleIndexEnd, thresholdIndex
            );
        }

        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        const std::vector<size_t>& sampleIndices_;
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
            const size_t, const std::tuple<int, int>&, const int = 0);
    void deserialize(std::istream&);

    size_t size() const;
    const DecisionTreeType& decisionTree(const size_t) const;
    void predict(const andres::View<Feature>&, andres::Marray<Probability>&, const std::vector<int>& = std::vector<int>{ff_numCores()}) const;
    void serialize(std::ostream&) const;
protected:
    std::vector<DecisionTreeType> decisionTrees_;

private:
    template<class RandomEngine>
       static void sampleBootstrap(const size_t, std::vector<size_t>&, RandomEngine&);
    
    
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
    return featureIndex_;
}

/// Returns, for a non-leaf node, the index of a feature wrt which a decision is made.
/// 
template<class FEATURE, class LABEL>
inline size_t& 
DecisionNode<FEATURE, LABEL>::featureIndex() {
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
    return label_;
}


/**
 * This function is a helper function for the learn() method.
 * It performs the parallel search for the optimal feature and threshold
 * to split the samples in the node. It parallelizes the search by using the map as the basic block.
 * It computes in parallel the Gini coefficients for each feature in numberOfFeaturesToBeAssessed
 * 
 * \param sampleIndexBegin Index of the first sample to be considered.
 * \param numberOfFeaturesToBeAssessed Number of features to be assessed for the split
 * \param numberOfClassesInSubset Number of classes in the subset of samples.
 * \param numberOfWorkers Number of workers to be used for parallel processing.
 * \param sortBuffer A buffer to hold the sorted feature values and sample indices.
 * \param featureIndicesBuffer A buffer to hold the indices of the features to be assessed.
 * \param sampleIndicesView A view of the sample indices to be considered.
 * \param features The features of the samples.
 * \param labels The labels of the samples.
 * \return A tuple containing the optimal sum of Gini coefficients, the index of the optimal
 * feature, the optimal threshold, and the index of the optimal threshold.
 * \addtogroup decision_node
 * \ingroup decision_tree
 * 
 */
template<class FEATURE, class LABEL>
template<std::ranges::range Inds>
inline std::tuple<double, size_t, FEATURE, size_t>
DecisionNode<FEATURE, LABEL>::helperParallelMap(
    const size_t& sampleIndexBegin,
    const size_t& numberOfFeaturesToBeAssessed, const size_t& numberOfClassesInSubset,
    const size_t& numberOfWorkers, std::vector<size_t>& featureIndicesBuffer, 
    Inds& sampleIndicesView,
    const andres::View<Feature>& features, const andres::View<Label>& labels
) {
    using namespace ff;
    
    BestSplit identity;
    ParallelForReduce<BestSplit> pfr (numberOfWorkers, MAP_SPINWAIT, MAP_SPINBARRIER);

    auto body = [&](const long j, BestSplit& bs_temp) {
        const size_t fi = featureIndicesBuffer[j];
        BestSplit bs_feature;
        bs_feature.featureIndex = fi;
        
        std::vector<SortItem<Feature>> localSortBuffer;
        const size_t numSamplesInNode = sampleIndicesView.size();
        localSortBuffer.reserve(numSamplesInNode);
        for (size_t k = 0; k < numSamplesInNode; ++k) {
            localSortBuffer.emplace_back(features(sampleIndicesView[k], fi), sampleIndicesView[k]);
        }
        std::sort(localSortBuffer.begin(), localSortBuffer.end());

        std::vector<size_t> numbersOfLabelsForSplit[2];
        numbersOfLabelsForSplit[0].assign(numberOfClassesInSubset, 0);
        numbersOfLabelsForSplit[1].assign(numberOfClassesInSubset, 0);
        
        size_t numbersOfElements[] = {0, numSamplesInNode};

        for(size_t k = 0; k < numSamplesInNode; ++k) {
            const Label label = labels(localSortBuffer[k].sampleIndex);
            ++numbersOfLabelsForSplit[1][label];
        }

        size_t thresholdIndexLoopVar = 0;
        for(;;){
            while(thresholdIndexLoopVar + 1 < numSamplesInNode
                && localSortBuffer[thresholdIndexLoopVar].featureValue == localSortBuffer[thresholdIndexLoopVar + 1].featureValue) {
                const Label label = labels(localSortBuffer[thresholdIndexLoopVar].sampleIndex);
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
                ++thresholdIndexLoopVar;
            }

            {
                const Label label = labels(localSortBuffer[thresholdIndexLoopVar].sampleIndex);
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
            }
            ++thresholdIndexLoopVar;
            if(thresholdIndexLoopVar == numSamplesInNode) {
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

            double giniCoefficients[2] = {0.0, 0.0};
            for(size_t s = 0; s < 2; ++s) {
                if(numbersOfElements[s] < 2) {
                    giniCoefficients[s] = 0.0;
                } else {
                    giniCoefficients[s] = static_cast<double>(numbersOfDistinctPairs[s])
                        / (static_cast<double>(numbersOfElements[s]) * (numbersOfElements[s] - 1));
                }
            }

            double sumOfGiniCoefficients = giniCoefficients[0] + giniCoefficients[1];
            double currentBalance = 
                        std::abs(static_cast<double>(numbersOfElements[0]) - static_cast<double>(numbersOfElements[1]))
                        / static_cast<double>(numbersOfElements[0] + numbersOfElements[1]);
            if(sumOfGiniCoefficients < bs_feature.gini ||
                (sumOfGiniCoefficients == bs_feature.gini && 
                    (currentBalance < bs_feature.balance || 
                        (currentBalance == bs_feature.balance && 
                            thresholdIndexLoopVar > bs_feature.relativeThresholdIndex)) )
            ) {
                bs_feature.balance = currentBalance;
                bs_feature.gini = sumOfGiniCoefficients;
                bs_feature.threshold = localSortBuffer[thresholdIndexLoopVar].featureValue;
                bs_feature.relativeThresholdIndex = thresholdIndexLoopVar;
            }
        }
        if(bs_feature.gini < bs_temp.gini ||
            (bs_feature.gini == bs_temp.gini && 
                (bs_feature.balance < bs_temp.balance || 
                    (bs_feature.balance == bs_temp.balance && 
                        bs_feature.relativeThresholdIndex > bs_temp.relativeThresholdIndex)) )
        ) {
            bs_temp = bs_feature;
        }
        return;
    };

    auto reduce = [](BestSplit& a, const BestSplit& b) {
        if(b < a) {
            a = b;
        }
    };

    BestSplit overallBestSplit;
    pfr.disableScheduler();
    pfr.parallel_reduce(
        overallBestSplit, identity, 0, numberOfFeaturesToBeAssessed, body, reduce
    );

    if(overallBestSplit.gini == std::numeric_limits<double>::infinity())
        return {overallBestSplit.gini, 0, Feature(), 0};
    
    std::vector<SortItem<Feature>> sortBuffer;
    const size_t numSamplesInNode = sampleIndicesView.size();
    sortBuffer.reserve(numSamplesInNode);
    for (size_t k = 0; k < numSamplesInNode; ++k) {
        sortBuffer.emplace_back(features(sampleIndicesView[k], overallBestSplit.featureIndex), sampleIndicesView[k]);
    }
    std::sort(sortBuffer.begin(), sortBuffer.end());

    for(size_t k = 0; k < numSamplesInNode; ++k) {
        sampleIndicesView[k] = sortBuffer[k].sampleIndex;
    }

    return {
        overallBestSplit.gini, 
        overallBestSplit.featureIndex, 
        overallBestSplit.threshold, 
        sampleIndexBegin + overallBestSplit.relativeThresholdIndex
    };
}


/**
 * This function is a helper function for the learn() method.
 * It performs the sequential search for the optimal feature and threshold
 * to split the samples in the node.
 * 
 * \param sampleIndicesView A view of the sample indices to be considered for the split.
 * \param numberOfFeaturesToBeAssessed Number of features to be assessed for the split.
 * \param sortBuffer A buffer to hold the sorted feature values and sample indices.
 * \param featureIndicesBuffer A buffer to hold the indices of the features to be assessed.
 * \param features The features of the samples.
 * \param labels The labels of the samples.
 * \return The optimal sum of Gini coefficients for the best split found.
 * 
 * \addtogroup decision_node
 * \ingroup decision_tree
 */
template<class FEATURE, class LABEL>
template<std::ranges::range Inds>
inline std::tuple<double, size_t, FEATURE, size_t> DecisionNode<FEATURE, LABEL>::helperSequential(
    Inds& sampleIndicesView, const size_t& sampleIndexBegin, const size_t& numberOfFeaturesToBeAssessed, const size_t& numberOfClassesInSubset,
    std::vector<SortItem<Feature>>& sortBuffer, std::vector<size_t>& featureIndicesBuffer,
    const andres::View<Feature>& features, const andres::View<Label>& labels
) {


    std::vector<size_t> numbersOfLabelsForSplit[2]; 
    numbersOfLabelsForSplit[0].reserve(10); 
    numbersOfLabelsForSplit[1].reserve(10); 
    double optimalSumOfGiniCoefficients = std::numeric_limits<double>::infinity();
    double bestBalance = std::numeric_limits<double>::infinity();
    size_t currentOptimalFeatureIndex = 0; // Initialize
    Feature currentOptimalThreshold = Feature(); // Initialize
    size_t currentOptimalThresholdIndex = sampleIndexBegin; // Initialize

    for(size_t j = 0; j < numberOfFeaturesToBeAssessed; ++j) {
        const size_t fi = featureIndicesBuffer[j];

        // To improve cache performance, create a temporary vector of structs
        // to hold feature values and indices, sort it, then update sampleIndices.
        // This makes the sort operation much faster by avoiding strided memory access.
       
        const auto numSamplesInNode = sampleIndicesView.size();

        std::sort(
            sampleIndicesView.begin(),
            sampleIndicesView.end(),
            ComparisonByFeature(features, fi)
        );
     

        #if defined(DEBUG)
            for(size_t k = 0; k + 1 < numSamplesInNode; ++k) {
                assert(
                    features(sampleIndicesView[k], fi) <= features(sampleIndicesView[k + 1], fi)
                );
            }
        #endif

        size_t numbersOfElements[] = {0, numSamplesInNode};
        
        for(size_t s = 0; s < 2; ++s) { // Clear/resize for current feature
            numbersOfLabelsForSplit[s].assign(numberOfClassesInSubset, 0);
        }

        for(size_t k = 0; k < numSamplesInNode; ++k) {
            const Label label = labels(sampleIndicesView[k]);
            // if(label >= numbersOfLabelsForSplit[1].size()) { // Should be handled by pre-sizing
            //      numbersOfLabelsForSplit[0].resize(label + 1, 0);
            //      numbersOfLabelsForSplit[1].resize(label + 1, 0);
            // }
            ++numbersOfLabelsForSplit[1][label];
        }

        std::ptrdiff_t thresholdIndexLoopVar = 0;
        for(;;) { 
            while(thresholdIndexLoopVar + 1 < numSamplesInNode
            && features(sampleIndicesView[thresholdIndexLoopVar], fi) 
            == features(sampleIndicesView[thresholdIndexLoopVar + 1], fi)) {                
                const Label label = labels(sampleIndicesView[thresholdIndexLoopVar]);
               
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
                ++thresholdIndexLoopVar;
            }

            {
                const Label label = labels(sampleIndicesView[thresholdIndexLoopVar]);
                
                ++numbersOfElements[0];
                --numbersOfElements[1];
                ++numbersOfLabelsForSplit[0][label];
                --numbersOfLabelsForSplit[1][label];
            }
            ++thresholdIndexLoopVar; 
            if(thresholdIndexLoopVar == numSamplesInNode) {
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
            double currentBalance = 
                std::abs(static_cast<double>(numbersOfElements[0]) - static_cast<double>(numbersOfElements[1]))
                / static_cast<double>(numbersOfElements[0] + numbersOfElements[1]);
            if(sumOfginiCoefficients < optimalSumOfGiniCoefficients ||
                (sumOfginiCoefficients == optimalSumOfGiniCoefficients && 
                    (currentBalance < bestBalance || 
                        (currentBalance == bestBalance && 
                            thresholdIndexLoopVar > currentOptimalThresholdIndex)) )
            ) {
                bestBalance = currentBalance;
                optimalSumOfGiniCoefficients = sumOfginiCoefficients;
                currentOptimalFeatureIndex = fi;
                currentOptimalThreshold = features(sampleIndicesView[thresholdIndexLoopVar], fi); 
                currentOptimalThresholdIndex = thresholdIndexLoopVar;
            }
        }
        for(size_t s = 0; s < 2; ++s) {
            std::fill(numbersOfLabelsForSplit[s].begin(), numbersOfLabelsForSplit[s].end(), 0);
        }
    }

    return {optimalSumOfGiniCoefficients, currentOptimalFeatureIndex, currentOptimalThreshold, currentOptimalThresholdIndex+ sampleIndexBegin};
}

/** 
 * @param features A matrix in which every row corresponds to a sample and every column corresponds to a feature.
 * @param labels A vector of labels, one for each sample.
 * @param sampleIndicesView A view of subset of indices of samples to be considered for learning. This vector is used by the function as a scratch-pad for sorting.
 * @param sampleIndexBegin Index of the first element of sampleIndicesView to be considered.
 * @param radomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
 */
template<class FEATURE, class LABEL>
template<std::ranges::range Inds>
size_t DecisionNode<FEATURE, LABEL>::learn(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    Inds& sampleIndicesView,
    const size_t& sampleIndexBegin,
    const int& randomSeed
) {
    #if defined(DEBUG)
    assert(sampleIndicesView.size() <= features.shape(0));
    assert(sampleIndicesView.size() != 0);
    #endif
    {
        bool isLabelUnique = true;
        const size_t firstLabelSampleIndex = sampleIndicesView.front();
        if( firstLabelSampleIndex >= labels.size()) {
            throw std::out_of_range("Sample index out of bounds for labels.");
        }
        const Label firstLabel = labels(firstLabelSampleIndex);
        for(std::ptrdiff_t ind = 0; ind < sampleIndicesView.size(); ++ind) {
            const size_t currentSampleIndex = sampleIndicesView[ind];
            if(currentSampleIndex >= labels.size()) {
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
            return 0; 
        }
    }

    const size_t numberOfFeatures = features.shape(1);
    assert(numberOfFeatures != 0);
    Label max_label = 0;
    for (std::ptrdiff_t ind = 0; ind < sampleIndicesView.size(); ++ind) {
        const size_t currentSampleIndex = sampleIndicesView[ind];
        if (labels(currentSampleIndex) > max_label) {
            max_label = labels(currentSampleIndex);
        }
    }
    const auto numberOfClassesInSubset = static_cast<size_t>(max_label) + 1;
    const size_t numberOfFeaturesToBeAssessed = 
        static_cast<size_t>(
            std::ceil(std::sqrt(
                static_cast<double>(numberOfFeatures)
            ))
    );
    std::vector<size_t> featureIndicesBuffer(numberOfFeaturesToBeAssessed);
    std::vector<SortItem<Feature>> sortBuffer(features.shape(0));
    std::vector<size_t> randomSampleBuffer;
    auto randomEngine = std::mt19937(randomSeed+sampleIndexBegin + 1);
    sampleSubsetWithoutReplacement(
        numberOfFeatures, 
        numberOfFeaturesToBeAssessed,
        featureIndicesBuffer,
        randomEngine,
        randomSampleBuffer
    );

    
#if USE_PARALLEL_MAP
    auto [optimalSumOfGiniCoefficients, currentOptimalFeatureIndex, currentOptimalThreshold, currentOptimalThresholdIndex] = 
        helperParallelMap(
            sampleIndexBegin,
            numberOfFeaturesToBeAssessed, numberOfClassesInSubset,
            MAP_NUM_WORKERS, featureIndicesBuffer,
            sampleIndicesView, features, labels
        );
#else
        auto [optimalSumOfGiniCoefficients, currentOptimalFeatureIndex, currentOptimalThreshold, currentOptimalThresholdIndex] = 
        helperSequential(sampleIndicesView, sampleIndexBegin, 
            numberOfFeaturesToBeAssessed, numberOfClassesInSubset, sortBuffer, featureIndicesBuffer,
            features, labels
        );
#endif
    if (optimalSumOfGiniCoefficients == std::numeric_limits<double>::infinity()) {
        isLeaf_ = true;
        std::unordered_map<Label, size_t> localLabelCounts;
        for (const auto& sampleIndex : sampleIndicesView)
            localLabelCounts[labels(sampleIndex)]++;

        Label mostFrequentLabel = labels(sampleIndicesView.front());

        size_t maxCount = 0;
        for (const auto& [label, count] : localLabelCounts)
            if (count > maxCount) {
                maxCount = count;
                mostFrequentLabel = label;
            }
        
        label_ = mostFrequentLabel;
        return 0;
    }

    this->featureIndex_ = currentOptimalFeatureIndex;
    this->threshold_ = currentOptimalThreshold;

    std::sort(
        sampleIndicesView.begin(),
        sampleIndicesView.end(),
        ComparisonByFeature(features, currentOptimalFeatureIndex)
    );

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
        #if defined(DEBUG)
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
    int randomSeed
) {
    assert(decisionNodes_.size() == 0);
   
    std::queue<TreeConstructionQueueEntry> queue;
    // learn root node
    {
        decisionNodes_.push_back(DecisionNodeType());
        auto sampleIndicesView = std::ranges::subrange(
            sampleIndices.begin(), 
            sampleIndices.end()
        );
        size_t thresholdIndex = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndicesView,
            0,
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
        auto sampleIndicesViewLeft = std::ranges::subrange(
            sampleIndices.begin() + sampleIndexBegin, 
            sampleIndices.begin() + thresholdIndex
        );
        decisionNodes_.push_back(DecisionNodeType());
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndicesViewLeft,
            sampleIndexBegin,
            randomSeed
        );
        #if defined(DEBUG)
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

        #if defined(DEBUG)
        assert(nodeIndexNew == decisionNodes_.size());
        assert(thresholdIndex < sampleIndexEnd);
        assert(thresholdIndex >= 0);
        assert(sampleIndexEnd <= sampleIndices.size());
        assert(sampleIndexEnd != sampleIndexBegin);
        assert(sampleIndexBegin < sampleIndexEnd);
        #endif

        decisionNodes_.push_back(DecisionNodeType());
        auto sampleIndicesViewRight = std::ranges::subrange(
            sampleIndices.begin() + thresholdIndex, 
            sampleIndices.begin() + sampleIndexEnd
        );
        thresholdIndexNew = decisionNodes_.back().learn(
            features, 
            labels, 
            sampleIndicesViewRight,
            thresholdIndex,
            randomSeed
        );
        #if defined(DEBUG)
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


/**
 * Learns the decision tree using FastFlow Farm for parallel node processing.
 * From experiments, it seems that parallelizing beyond this network does not help much.
 * To note: the most critical part (computation of the Gini coefficients for the best split), see helperParallelMap, is parallelized using ParallelFor, which i suspect recreates the threads at each call.
 * The final version is using the helperSequential function, which is faster in practice. 
 * The fastflow Network is the following:
 *              |               |
 *              |   workers     |
 *              |               |
 * sourceSink ->|   workers     | -> sourceSink (same as input to wrap around)
 *              |               | 
 *              |   workers     |
 *              |               |
 *              |   workers     |
 *              |               |
 *              |    ....       |
 *              |               |
 */             

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
                    std::make_unique<Worker>(sourceSink.features_, sourceSink.labels_, sourceSink.sampleIndices_, randomSeed)
                );
            }
            return std::move(workers);
        } (),
        sourceSink
    );
    
    farm.remove_collector(); 
    farm.wrap_around();

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
                #if defined(DEBUG)
                assert(fi < numberOfFeatures);
                #endif
                if(features(j, fi) < threshold) {
                    nodeIndex = decisionNode.childNodeIndex(0);
                }
                else {
                    nodeIndex = decisionNode.childNodeIndex(1);
                }
                #if defined(DEBUG)
                assert(nodeIndex != 0); // assert that tree is not incomplete
                #endif
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


// Learns a forest using the farm-based learning paradigm.
// The FastFlow network is the following:
//              |               |               |
//              |               |  workers      |
//              |               |               |
// farmWork1  ->| sourceSink    |  workers      |
//              |               |               |
//              |               |  workers      |
//              |               |               |
//              |               |  workers      |
//              |               |               |
//              |               |    ....       |
//
//
//              |               |               |
//              |               |               |
// farmWork2  ->|               |  workers     |
//              |               |               |
//              | sourceSink    |  workers      |
//              |               |               |
//              |               |  workers      |
//              |               |               |
//              |               |  workers      |
//              |               |    ....       |
//        ....................................



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
    const std::tuple<int, int>& workersConfig, // (farmWorkers, workersPerTree)
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
    const auto [farmWorkers, workersPerTree] = workersConfig;
    std::cout << "Learning decision forest with " << numberOfDecisionTrees 
              << " trees, each using a farm with " << workersPerTree 
              << " workers, and a total of " << farmWorkers << " farm workers." << std::endl;
              
    clear();
    decisionTrees_.resize(numberOfDecisionTrees);
    // Use the number of workers for the forest as determined by the estimator.
    ff::ParallelFor pf(farmWorkers, FOR_SPINWAIT, FOR_SPINBARRIER);
    pf.blocking_mode();
    pf.parallel_for(
        0, static_cast<ptrdiff_t>(decisionTrees_.size()), 1, static_cast<int>(std::copysignf(1.0, static_cast<float>(FOR_CHUNK_SIZE)))*std::min(std::abs(FOR_CHUNK_SIZE), static_cast<int>(numberOfDecisionTrees / farmWorkers)),
        [&](ptrdiff_t treeIndex) {
            std::vector<size_t> sampleIndices(numberOfSamples);
            
            std::mt19937 randomEngine;
            auto tree_specific_seed = randomSeed + treeIndex;
            if(randomSeed == 0){
                std::seed_seq seedSeq{static_cast<unsigned int>(std::random_device{}()), static_cast<unsigned int>(treeIndex)};
                randomEngine.seed(seedSeq);
            } else {
                randomEngine.seed(tree_specific_seed);
            }
            
            
            sampleBootstrap(numberOfSamples, sampleIndices, randomEngine);
            // Choose learning strategy based on nested parallelism decision
            decisionTrees_[treeIndex].learnWithFarm(
                features, labels, sampleIndices, tree_specific_seed, workersPerTree
            );
            
        }, farmWorkers
    ); 
}


// Predicts label probabilities using FastFlow ParallelForReduce.

/// Predict the label probabilities of samples as described by Breiman (2001).
///
/// \param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
/// \param labelProbabilities A matrix of probabilities in which every rows corresponds to a sample and every column corresponds to a label.
///
template<class FEATURE, class LABEL, class PROBABILITY>
inline void 
DecisionForest<FEATURE, LABEL, PROBABILITY>::predict(
    const andres::View<Feature>& features,
    andres::Marray<Probability>& labelProbabilities,
    const std::vector<int>& nWorkers 
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
    const size_t numberOfLabels  = labelProbabilities.shape(1);


    using ProbArray = andres::Marray<Probability>;
    // create a “zero” identity array
    const size_t shape[] = {numberOfSamples, numberOfLabels};
    ProbArray identity(shape, shape+2);
    std::fill(identity.begin(), identity.end(), Probability());
    std::fill(labelProbabilities.begin(), labelProbabilities.end(), Probability());

    // set up the ParallelForReduce
    ff::ParallelForReduce<ProbArray> pfr(nWorkers[0], FOR_SPINWAIT, FOR_SPINBARRIER);

    // body: for each tree, predict & increment local counts
    auto body = [&](long t, ProbArray & local) {
        std::vector<Label> labels(numberOfSamples);
        decisionTrees_[t].predict(features, labels);
        for(size_t i = 0; i < numberOfSamples; ++i)
            local(i, labels[i]) += Probability(1);
    };

    // reduction: element‐wise add
    auto reduce = [&](ProbArray & acc, const ProbArray & val) {
        auto a = acc.begin();
        auto b = val.begin();
        for(; a != acc.end(); ++a, ++b) {
            *a += *b;
        }
    };

    // run the parallel_reduce over [0, nTrees):
    pfr.parallel_reduce(
      labelProbabilities,  // result
      identity,            // init value for each worker
      0, 
      static_cast<ptrdiff_t>(decisionTrees_.size()),
      body,
      reduce
    );

    // finally normalize to get probabilities
    for(auto & v : labelProbabilities) {
        v /= decisionTrees_.size();
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

#endif //