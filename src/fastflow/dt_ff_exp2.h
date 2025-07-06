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
#include <unordered_map>
#include <tuple>
#include <immintrin.h>
#include <thread>
#include <variant>


#include "ff/ff.hpp"
#include "ff/parallel_for.hpp"
#include "ff/farm.hpp"
#include "ff/map.hpp"

#include "ff_impl_config.h"
 

#include "marray.h"

#define LEFT_NODE 0
#define RIGHT_NODE 1
#define ROOT_NODE 2

#define SKIP_RA2A_WORKERS true
#define NEW_TREE true
#define ORDER_INDICES true

/// The public API.
namespace andres {
    
/// Machine Learning.
namespace ff_ml_exp2 {
    template<typename F>
    struct SortItem {
        F featureValue;
        size_t sampleIndex;
        bool operator<(const SortItem& other) const {
                return featureValue < other.featureValue;
        }
    };
    
class OptimizedGiniCalculator {
public:
    // Calculates sum_{i<j} counts[i]*counts[j] using the identity:
    // sum_{i<j} c_i*c_j = ( (sum c_i)^2 - sum(c_i^2) ) / 2
    static size_t computeDistinctPairs(const std::vector<size_t>& labelCounts, size_t totalSamples) {
        if (totalSamples < 2) {
            return 0;
        }
        
        
        __m256d sum_sq_vec = _mm256_setzero_pd();
        size_t i = 0;
        
        for (; i + 4 <= labelCounts.size(); i += 4) {
            
            double vals[4] = {
                static_cast<double>(labelCounts[i]), static_cast<double>(labelCounts[i+1]),
                static_cast<double>(labelCounts[i+2]), static_cast<double>(labelCounts[i+3])
            };
            __m256d counts_pd = _mm256_loadu_pd(vals);

    
            sum_sq_vec = _mm256_add_pd(sum_sq_vec, _mm256_mul_pd(counts_pd, counts_pd));
        }

        alignas(32) double sum_sq_parts[4];
        _mm256_store_pd(sum_sq_parts, sum_sq_vec);
        double sum_of_squares = sum_sq_parts[0] + sum_sq_parts[1] + sum_sq_parts[2] + sum_sq_parts[3];

        for (; i < labelCounts.size(); ++i) {
            sum_of_squares += static_cast<double>(labelCounts[i]) * static_cast<double>(labelCounts[i]);
        }

        double totalSamples_d = static_cast<double>(totalSamples);
        double total_pairs_d = ((totalSamples_d * totalSamples_d) - sum_of_squares) / 2.0;
        
        return static_cast<size_t>(total_pairs_d + 0.5);
    }
};

template<class FEATURE, class LABEL, class PROBABILITY>
class DecisionForest;

/// A node in a decision tree.
template<class FEATURE, class LABEL>
class DecisionNode {
    public:
        struct BestSplit;

        typedef FEATURE Feature;
        typedef LABEL Label;
        typedef DecisionForest<FEATURE, LABEL, double>::FEN_out_t LA2AN_in_t;
        typedef DecisionForest<FEATURE, LABEL, double>::FEN_in_t FCN_out_t;
        using VLA2A_out_t = std::tuple<LA2AN_in_t*, size_t, size_t>;
        using LA2AN_out_t = std::variant<FCN_out_t, VLA2A_out_t>;
        using VRA2A_out_t = std::tuple<LA2AN_in_t*, size_t, std::unique_ptr<BestSplit>>;
        using RA2AN_out_t = std::variant<FCN_out_t, VRA2A_out_t>;
        typedef LA2AN_out_t RA2AN_in_t;
 
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

    struct WorkerRA2A: ff::ff_minode_t<RA2AN_in_t, RA2AN_out_t>{
        
        WorkerRA2A(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            const std::vector<std::vector<size_t>>& matrixSampleIndices
        ): 
            features_(features),
            labels_(labels),
            matrixSampleIndices_(matrixSampleIndices),
            sortBuffer_(features_.shape(0)),
            numberOfLabelsForSplit(2)
        {}

        RA2AN_out_t* svc(RA2AN_in_t* task) {
            if (std::holds_alternative<FCN_out_t>(*task)) {
                this->ff_send_out(task);
                return this->GO_ON;
            } else {
                auto& [prevTask, numberOfClassesInSubset, fi] = std::get<VLA2A_out_t>(*task);
                auto& [treeIndex, nodeType, parentIndex, childIndex, sampleIndexBegin, sampleIndexEnd] = *prevTask;
                auto bs = std::make_unique<BestSplit>();
                const auto& subSampleIndices = std::ranges::subrange(
                    matrixSampleIndices_[treeIndex].begin() + sampleIndexBegin,
                    matrixSampleIndices_[treeIndex].begin() + sampleIndexEnd
                );
                
                const auto numSamplesInNode = subSampleIndices.size();
#ifdef DEBUG
                assert(numSamplesInNode <= features_.shape(0));
                assert(numSamplesInNode <= labels_.shape(0));
                assert(fi < features_.shape(1));
                assert(numSamplesInNode > 0);
#endif
                
                for(size_t ind = 0; ind < numSamplesInNode; ++ind) {
                    // Prefetch data for the next few iterations
                    if (ind + PREFETCH_SAMPLES < numSamplesInNode) {
                        _mm_prefetch((const char*)subSampleIndices[ind + PREFETCH_SAMPLES], _MM_HINT_T0);
                        _mm_prefetch((const char*)&features_(subSampleIndices[ind + PREFETCH_SAMPLES], fi), _MM_HINT_T0);
                    }
                    sortBuffer_[ind] = std::move(SortItem<Feature>(features_(subSampleIndices[ind], fi), subSampleIndices[ind]));
                }

                std::sort(sortBuffer_.begin(), sortBuffer_.begin() + numSamplesInNode);

                
                numberOfLabelsForSplit[0].resize(numberOfClassesInSubset, 0);
                numberOfLabelsForSplit[1].resize(numberOfClassesInSubset, 0);

                std::vector numberOfElements {0, numSamplesInNode};

                for(size_t k = 0; k < numSamplesInNode; ++k)
                    ++numberOfLabelsForSplit[1][labels_(sortBuffer_[k].sampleIndex)];

                size_t thresholdIndexLoopVar = 0;

                for(;;){

                    while(thresholdIndexLoopVar + 1 < numSamplesInNode &&
                          sortBuffer_[thresholdIndexLoopVar].featureValue == sortBuffer_[thresholdIndexLoopVar + 1].featureValue) {
                        ++numberOfElements[0];
                        --numberOfElements[1];

                        auto label = labels_(sortBuffer_[thresholdIndexLoopVar].sampleIndex);
                        ++numberOfLabelsForSplit[0][label];
                        --numberOfLabelsForSplit[1][label];
                        ++thresholdIndexLoopVar;
                    }

                    {
                        const Label label = labels_(sortBuffer_[thresholdIndexLoopVar].sampleIndex);
                        ++numberOfElements[0];
                        --numberOfElements[1];
                        ++numberOfLabelsForSplit[0][label];
                        --numberOfLabelsForSplit[1][label];
                    }
                    ++thresholdIndexLoopVar;
                    if(thresholdIndexLoopVar >= numSamplesInNode) 
                        break;
            
#if defined(DEBUG)
                    assert(thresholdIndexLoopVar < numSamplesInNode);
                    assert(sortBuffer_[thresholdIndexLoopVar - 1].featureValue <= sortBuffer_[thresholdIndexLoopVar].featureValue);
                    assert(numberOfLabelsForSplit[0].size() == numberOfLabelsForSplit[1].size());
#endif
                    std::vector numberOfDistinctPairs {0, 0};
                    for(size_t s = 0; s < 2; ++s) {
                        for(size_t k_label = 0; k_label < numberOfLabelsForSplit[s].size(); ++k_label) {
                            for(size_t m_label = k_label + 1; m_label < numberOfLabelsForSplit[s].size(); ++m_label) {
                                numberOfDistinctPairs[s] += 
                                    numberOfLabelsForSplit[s][k_label] * numberOfLabelsForSplit[s][m_label];
                            }
                        }   
                    }

                    std::vector giniCoefficient {0.0, 0.0};
                    for(size_t s = 0; s < 2; ++s) 
                        if(numberOfElements[s] < 2){
                            giniCoefficient[s] = 0.0;
                        } else {
                            giniCoefficient[s] = 
                                static_cast<double>(numberOfDistinctPairs[s]) /
                                static_cast<double>(numberOfElements[s] * (numberOfElements[s] - 1));
                        }
                    auto sumGiniCoefficient = giniCoefficient[0] + giniCoefficient[1];
                    double currentBalance = 
                        std::abs(static_cast<double>(numberOfElements[0]) - static_cast<double>(numberOfElements[1]))
                        / static_cast<double>(numberOfElements[0] + numberOfElements[1]);
                    if(sumGiniCoefficient < bs->gini ||
                          (sumGiniCoefficient == bs->gini && currentBalance < bs->balance)
                    ) {
                        bs->balance = currentBalance;
                        bs->gini = sumGiniCoefficient;
                        bs->featureIndex = fi;
                        bs->threshold = sortBuffer_[thresholdIndexLoopVar].featureValue;
                        bs->relativeThresholdIndex = thresholdIndexLoopVar;
                    }
                }
                this->ff_send_out(
                    new RA2AN_out_t(
                        std::make_tuple(
                            prevTask,
                            numberOfClassesInSubset,
                            std::move(bs)
                        )
                    
                    )
                );
                delete task;
                return this->GO_ON;   
            }
            
        }


        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        const std::vector<std::vector<size_t>>& matrixSampleIndices_;
        std::vector<SortItem<Feature>> sortBuffer_;
        std::vector<std::vector<size_t>> numberOfLabelsForSplit;

    };

    struct WorkerLA2A: ff::ff_monode_t<LA2AN_in_t, LA2AN_out_t>{

        WorkerLA2A(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            const size_t numberOfFeaturesToBeAssessed,
            const std::vector<std::vector<size_t>>& matrixSampleIndices,
            const int& randomSeed
        ): 
            features_(features),
            labels_(labels),
            numberOfFeaturesToBeAssessed_(numberOfFeaturesToBeAssessed),
            matrixSampleIndices_(matrixSampleIndices),
            randomSeed_(randomSeed),
            featureIndicesBuffer_( numberOfFeaturesToBeAssessed_),
            randomSamplerBuffer_(features.shape(1))
        {}

        LA2AN_out_t* svc(LA2AN_in_t* task) {
            auto& [treeIndex, nodeType, parentIndex, childIndex, sampleIndexBegin, sampleIndexEnd] = *task;
            const auto& subSampleIndices = std::ranges::subrange(
                matrixSampleIndices_[treeIndex].begin() + sampleIndexBegin,
                matrixSampleIndices_[treeIndex].begin() + sampleIndexEnd
            );

            bool isLabelUnique = true;
            const size_t firstLabelIndex = subSampleIndices[0];
#ifdef DEBUG
            assert(firstLabelIndex < labels_.shape(0));
            assert(subSampleIndices.size() == sampleIndexEnd-sampleIndexBegin);
#endif
            const Label firstLabel = labels_(firstLabelIndex);
            for(size_t ind = 1; ind < subSampleIndices.size(); ++ind) {
                const size_t labelIndex = subSampleIndices[ind];
#ifdef DEBUG
                assert(labelIndex < labels_.shape(0));
#endif
                if(labels_(labelIndex) != firstLabel) {
                    isLabelUnique = false;
                    break;
                }
            }

            if(isLabelUnique) {
                auto newNode = std::make_unique<DecisionNode<Feature, Label>>();
                newNode->isLeaf() = true;
                newNode->label() = firstLabel;
                
                FCN_out_t fdn_out_task = std::make_pair(
                    std::make_tuple(
                        !NEW_TREE, treeIndex,
                        std::move(newNode), nodeType, parentIndex, childIndex, 
                        sampleIndexBegin, sampleIndexEnd, 0
                    ), 
                    !ORDER_INDICES
                );
                
                this->ff_send_out(
                    new LA2AN_out_t(
                        std::move(fdn_out_task) // Move the FCN_out_t into the variant
                    )
                );
                delete task;
                return this->GO_ON;
            }
            
            Label maxLabel = 0;
            for(size_t ind = 0; ind < subSampleIndices.size(); ++ind)
                if(labels_(subSampleIndices[ind]) > maxLabel) {
                    maxLabel = labels_(subSampleIndices[ind]);
                }
            
            const auto numberOfClassesInSubset = static_cast<size_t>(maxLabel) + 1;

            
            
            auto randomEngie = std::mt19937((randomSeed_ == 0) ? 0 : (randomSeed_ + sampleIndexBegin + treeIndex));

            sampleSubsetWithoutReplacement(
                features_.shape(1), numberOfFeaturesToBeAssessed_,
                featureIndicesBuffer_, randomEngie, randomSamplerBuffer_
            );
            
            for(size_t ind = 0; ind < numberOfFeaturesToBeAssessed_; ++ind) {
                
                this->ff_send_out(
                    new LA2AN_out_t(
                        std::make_tuple(
                            task, numberOfClassesInSubset, featureIndicesBuffer_[ind]
                        )
                    )
                );
            }
            return this->GO_ON;
        
        }

        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        const size_t numberOfFeaturesToBeAssessed_;
        const int randomSeed_;
        std::vector<size_t> featureIndicesBuffer_;
        std::vector<size_t> randomSamplerBuffer_;
        const std::vector<std::vector<size_t>>& matrixSampleIndices_;


    };
        
};

/// A decision tree.
template<class FEATURE = double, class LABEL = unsigned char>
class DecisionTree {
    
public:
    typedef FEATURE Feature;
    typedef LABEL Label;
    typedef DecisionNode<Feature, Label> DecisionNodeType;

    using task_t = std::tuple<int, size_t, size_t, size_t, size_t, std::vector<size_t>>;
    using in_t = std::tuple<DecisionNodeType, int, size_t, size_t, size_t, size_t, size_t, std::vector<size_t>>;


    DecisionTree();
    void deserialize(std::istream&);
    size_t size() const; // number of decision nodes
    void predict(const andres::View<Feature>&, std::vector<Label>&) const;
    const DecisionNodeType& decisionNode(const size_t) const;
    void serialize(std::ostream&) const;
    DecisionNodeType& operator[](const size_t);
    void addEmptyNode();
    bool isEmpty() const;
    size_t sizeTrees() const;


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

    using FEN_in_t = std::pair<std::tuple<decltype(NEW_TREE), size_t, std::unique_ptr<typename DecisionTreeType::DecisionNodeType>, int, size_t, size_t, size_t, size_t, size_t>, decltype(ORDER_INDICES)>;
    using FEN_out_t = std::tuple<size_t, int, size_t, size_t, size_t, size_t>;
    
    typedef FEN_in_t FCN_out_t;
    typedef DecisionTreeType::DecisionNodeType::RA2AN_out_t FCN_in_t;

    DecisionForest();
    void clear();    
    void learnWithFFNetwork(
        const andres::View<Feature>&,
        const andres::View<Label>&,
        const size_t&,
        const int = 0
    );
    void deserialize(std::istream&);

    size_t size() const;
    const DecisionTreeType& decisionTree(const size_t) const;
    void predict(const andres::View<Feature>&, andres::Marray<Probability>&) const;
    void serialize(std::ostream&) const;

private:
    template<class RandomEngine>
       static void sampleBootstrap(const size_t, std::vector<size_t>&, RandomEngine&);

    std::vector<DecisionTreeType> decisionTrees_;
    
    struct TupleHasher {
        template <class T>
        inline void hash_combine(std::size_t& seed, const T& v) const {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        std::size_t operator()(const std::tuple<size_t, decltype(LEFT_NODE), size_t, size_t>& k) const {
            std::size_t seed = 0;
            hash_combine(seed, std::get<0>(k));
            hash_combine(seed, std::get<1>(k));
            hash_combine(seed, std::get<2>(k));
            hash_combine(seed, std::get<3>(k));
            return seed;
        }
    };
    using MapType = std::unordered_map<
        std::tuple<size_t, decltype(LEFT_NODE), size_t, size_t>,
        std::pair<size_t, std::unique_ptr<typename DecisionTreeType::DecisionNodeType::BestSplit>>,
        TupleHasher
    >;
    struct TrainEmitter: ff::ff_monode_t<FEN_in_t, FEN_out_t> {
        
        TrainEmitter(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            std::vector<DecisionTreeType>& decisionTrees,
            const int& randomSeed = 0
        ):
            features_(features),
            labels_(labels),
            decisionTrees_(decisionTrees),
            randomSeed_(randomSeed),
            matrixSampleIndices_(
                decisionTrees.size(), 
                std::vector<size_t>(features.shape(0))
            ),
            queues_(decisionTrees.size(), std::queue<typename DecisionTreeType::TreeConstructionQueueEntry>())
        {
        }


        void init_tree(size_t treeIndex){
#if defined(DEBUG)
                assert(treeIndex < decisionTrees_.size());
                assert(decisionTrees_[treeIndex].isEmpty());
#endif          

            decisionTrees_[treeIndex].addEmptyNode();
#if defined(DEBUG)
            assert(!decisionTrees_[treeIndex].isEmpty());
#endif      
            std::mt19937 randomEngine_;
            if(randomSeed_ == 0){
                std::seed_seq seedSeq{static_cast<unsigned int>(std::random_device{}()), static_cast<unsigned int>(treeIndex)};
                randomEngine_.seed(seedSeq);
            } else {
                randomEngine_.seed(randomSeed_ + treeIndex);
            }
            sampleBootstrap(
                features_.shape(0), 
                matrixSampleIndices_[treeIndex], 
                randomEngine_
            );
        
            this->ff_send_out(
                new FEN_out_t(
                    treeIndex, ROOT_NODE, 0, 0, 0, matrixSampleIndices_[treeIndex].size()
                )
            );
            
            tasksInFlight_ += 1; // Increment the task count for the new tree.

        }
        
        FEN_out_t* svc(FEN_in_t* task){
            
                
            auto& [new_tree,treeIndex, newNode, nodeType, nodeIndexParent, nodeIndexChild, sampleIndexBegin, sampleIndexEnd, thresholdIndex] = task->first;
            auto orderSampleIndices = task->second;

            if(orderSampleIndices){
                sortItems_.resize(sampleIndexEnd - sampleIndexBegin);
                for(size_t ind = 0; ind < sortItems_.size(); ++ind) {
                    // Prefetch data for the next few iterations
                    if (ind + PREFETCH_SAMPLES < sortItems_.size()) {
                        _mm_prefetch((const char*)&matrixSampleIndices_[treeIndex][ind+sampleIndexBegin+PREFETCH_SAMPLES], _MM_HINT_T0);
                        _mm_prefetch((const char*)&features_(matrixSampleIndices_[treeIndex][ind+sampleIndexBegin+PREFETCH_SAMPLES], newNode->featureIndex()), _MM_HINT_T0);
                    }
                    sortItems_[ind] = std::move(SortItem<Feature>{
                        features_(matrixSampleIndices_[treeIndex][ind+sampleIndexBegin], newNode->featureIndex()),
                        matrixSampleIndices_[treeIndex][ind+sampleIndexBegin]
                    });
                    
                }
                std::sort(sortItems_.begin(), sortItems_.end());
                for(size_t ind = 0; ind < sortItems_.size(); ++ind)
                    matrixSampleIndices_[treeIndex][ind+sampleIndexBegin] = sortItems_[ind].sampleIndex;


            }
           
            if(new_tree){
                init_tree(treeIndex);
            } else {
                if(nodeType == ROOT_NODE) {
                    
                    decisionTrees_[treeIndex][0].featureIndex() = newNode->featureIndex();
                    decisionTrees_[treeIndex][0].threshold() = newNode->threshold();
                    decisionTrees_[treeIndex][0].isLeaf() = newNode->isLeaf();
                    decisionTrees_[treeIndex][0].label() = newNode->label();
                    
                } else {
                    // Process the task for an existing tree.
                    decisionTrees_[treeIndex][nodeIndexChild] = std::move(*newNode);
                    if(nodeType == LEFT_NODE) {
                        decisionTrees_[treeIndex][nodeIndexParent].childNodeIndex(LEFT_NODE) = nodeIndexChild;
                    } else if(nodeType == RIGHT_NODE) {
                        decisionTrees_[treeIndex][nodeIndexParent].childNodeIndex(RIGHT_NODE) = nodeIndexChild;
                    }
                }
                
                if(!decisionTrees_[treeIndex][nodeIndexChild].isLeaf()) {
#if defined(DEBUG)
                    assert(sampleIndexBegin <= thresholdIndex && thresholdIndex < sampleIndexEnd);
#endif
                    queues_[treeIndex].emplace(nodeIndexChild, sampleIndexBegin, sampleIndexEnd, thresholdIndex);
                }
            }

            
            if(!queues_[treeIndex].empty()) {
                auto entry = queues_[treeIndex].front();
                queues_[treeIndex].pop();
                auto nodeIndexNewLeft = decisionTrees_[treeIndex].sizeTrees();
                decisionTrees_[treeIndex].addEmptyNode();

                auto nodeIndexNewRight = decisionTrees_[treeIndex].sizeTrees();
                decisionTrees_[treeIndex].addEmptyNode();

#if defined(DEBUG)
                assert(entry.sampleIndexBegin_ < entry.sampleIndexEnd_);
                assert(entry.thresholdIndex_ > entry.sampleIndexBegin_);
                assert(entry.thresholdIndex_ < entry.sampleIndexEnd_);
#endif            
                
                this->ff_send_out(
                    new FEN_out_t(
                        treeIndex, LEFT_NODE, entry.nodeIndex_, nodeIndexNewLeft, entry.sampleIndexBegin_, 
                        entry.thresholdIndex_
                ));

                this->ff_send_out(
                    new FEN_out_t(
                        treeIndex, RIGHT_NODE, entry.nodeIndex_, nodeIndexNewRight, entry.thresholdIndex_, 
                        entry.sampleIndexEnd_
                    ));
                tasksInFlight_ += 2;
            }

            
            delete task;
            if(this->get_channel_id() == 0){
                --tasksInFlight_;
                if(tasksInFlight_ <= 0 && eosreceived_) {
#if defined(DEBUG)
                for(auto& queue: queues_)
                    assert(queue.empty() && "All queues should be empty at the end of processing.");             
#endif
                    return this->EOS; // Signal end of stream.
                }
            }
            return this->GO_ON; // Continue processing.

            
        }
        
        void eosnotify(ssize_t id) {
            // Notify the end of the stream.
            eosreceived_ = true;
            if(tasksInFlight_ == 0)
                this->broadcast_task(this->EOS);
               
        }
        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        std::vector<DecisionTreeType>& decisionTrees_;
        const int randomSeed_;

        std::vector<std::vector<size_t>> matrixSampleIndices_;
        std::vector<SortItem<Feature>> sortItems_;
        
        std::vector<std::queue<typename DecisionTreeType::TreeConstructionQueueEntry>> queues_;
        int tasksInFlight_ = 0;
        bool eosreceived_ = false; // Flag to indicate if EOS has been received.
    };

    struct TrainCollector: ff::ff_monode_t<FCN_in_t, FCN_out_t> {
        TrainCollector(
            const andres::View<Feature>& features,
            const andres::View<Label>& labels,
            const size_t numberOfFeaturesToBeAssessed,
            const std::vector<std::vector<size_t>>& matrixSampleIndices
        ): 
            features_(features),
            labels_(labels),
            numberOfFeaturesToBeAssessed_(numberOfFeaturesToBeAssessed),
            matrixSampleIndices_(matrixSampleIndices)
        {}

        FCN_out_t* svc(FCN_in_t* task) {
            if(std::holds_alternative<FCN_out_t>(*task)) {
                this->ff_send_out(task);
                return this->GO_ON;
            } else {
                auto& [prevTask, numberOfClassesInSubset, splitObj] = std::get<typename DecisionTreeType::DecisionNodeType::VRA2A_out_t>(*task);
                auto& [treeIndex, nodeType, parentIndex, childIndex, sampleIndexBegin, sampleIndexEnd] = *prevTask;
                auto key = std::make_tuple(
                    treeIndex, nodeType, parentIndex, childIndex
                );
                
                const auto& subSampleIndices = std::ranges::subrange(
                    matrixSampleIndices_[treeIndex].begin() + sampleIndexBegin,
                    matrixSampleIndices_[treeIndex].begin() + sampleIndexEnd
                );
                if(auto it = countsFeatures_.find(key); it != countsFeatures_.end()) {
                    ++it->second.first;
                    if(it->second.second->gini > splitObj->gini ||
                       (it->second.second->gini == splitObj->gini && it->second.second->balance > splitObj->balance)
                    ) {

                        it->second.second = std::move(splitObj);
                    }

                    if(it->second.first == numberOfFeaturesToBeAssessed_) {
                      
                        auto newNode = std::make_unique<DecisionNode<Feature, Label>>();
                        if(it->second.second->gini == std::numeric_limits<double>::infinity()) {

                            // No valid split found, create a leaf node
                            newNode->isLeaf() = true;
                            labelCounts_.resize(numberOfClassesInSubset);
                            std::fill(labelCounts_.begin(), labelCounts_.end(), 0);
                            for(const auto& sampleIndex : subSampleIndices)
                                ++labelCounts_[labels_(sampleIndex)];
                            
                            auto max_it = std::max_element(labelCounts_.begin(), labelCounts_.end());
                            newNode->label() = std::distance(labelCounts_.begin(), max_it);

                            auto result = std::make_pair(
                                std::make_tuple(
                                    !NEW_TREE, treeIndex,
                                    std::move(newNode), nodeType, parentIndex, childIndex,
                                    sampleIndexBegin, sampleIndexEnd, 0
                                ), 
                                !ORDER_INDICES
                            );
                            this->ff_send_out(new FCN_out_t(std::move(result)));
                        } else {
                        
                            auto bestSplit = std::move(it->second.second);
                            
                            newNode->isLeaf() = false;
                            newNode->featureIndex() = bestSplit->featureIndex;
                            newNode->threshold() = bestSplit->threshold;

                            FCN_out_t fdn_out_task = std::make_pair(
                                std::make_tuple(
                                    !NEW_TREE, treeIndex,
                                    std::move(newNode), nodeType, parentIndex, childIndex,
                                    sampleIndexBegin, sampleIndexEnd, bestSplit->relativeThresholdIndex+sampleIndexBegin
                                ), 
                                ORDER_INDICES
                            );

                            this->ff_send_out(
                                new FCN_out_t(std::move(fdn_out_task))
                            );
                            
                        }
                        countsFeatures_.erase(it);
                        delete prevTask;
                        
                    }
                } else {
                    countsFeatures_.emplace(
                        key, 
                        std::make_pair(1, std::move(splitObj))
                    );
                }

              
                
                
                delete task;
                return this->GO_ON;
            }
        
        }


        const andres::View<Feature>& features_;
        const andres::View<Label>& labels_;
        const size_t numberOfFeaturesToBeAssessed_;
        const std::vector<std::vector<size_t>>& matrixSampleIndices_;

        MapType countsFeatures_;
        std::vector<size_t> labelCounts_;
    };

    struct TrainGenerator: ff::ff_monode_t<FEN_in_t> {
        TrainGenerator(const long numTrees): 
            numTrees_(numTrees),
            currentTreeIndex_(0)
        {}
        FEN_in_t* svc(FEN_in_t*) {
            if(currentTreeIndex_ < numTrees_) {
                auto task = new FEN_in_t(
                    std::make_tuple(
                        NEW_TREE, currentTreeIndex_,
                        std::make_unique<typename DecisionTreeType::DecisionNodeType>(),
                        LEFT_NODE, 0, 0, 0, 0, 0
                    ),
                    !ORDER_INDICES
                );
                this->ff_send_out(task);
                ++currentTreeIndex_;
                return this->GO_ON; // Continue processing.
            } else {
                return this->EOS; // No more trees to generate.
            }
        }
                

                
        const long numTrees_;
        long currentTreeIndex_;
    };


    using PWN_out_t = std::tuple<size_t, size_t, std::vector<Label>>;
    using PEN_out_t = std::tuple<size_t, size_t, size_t>;
    typedef PEN_out_t PWN_in_t;

    struct PredWorker: ff::ff_node_t<PWN_in_t, PWN_out_t> {

        PredWorker(
            const andres::View<Feature>& features,
            const std::vector<DecisionTreeType>& decisionTrees_
        ):
            features_(features),
            decisionTrees_(decisionTrees_)
        {}

        PWN_out_t* svc(PWN_in_t* task) {
            auto& [samplesBegin, samplesEnd, indexTree] = *task;
            auto& tree = decisionTrees_[indexTree];
#if defined(DEBUG)
            assert(samplesBegin < samplesEnd);
            assert(indexTree < decisionTrees_.size());
            assert(samplesEnd <= features_.shape(0));
            assert(samplesBegin < features_.shape(0));
#endif
            constexpr size_t BATCH = static_cast<size_t>(BATCH_SAMPLES_TO_PREDICT);
            std::vector<Label> labels(samplesEnd - samplesBegin);

            for(size_t base = samplesBegin; base < samplesEnd; base += BATCH) {
                const size_t len = std::min(BATCH, samplesEnd - base);

                size_t nodeIdx[BATCH];
                bool   done[BATCH] = {false};

                for(size_t b = 0; b < len; ++b) nodeIdx[b] = 0;

                bool allDone = false;
                while(!allDone) {
                    allDone = true;
                    for(size_t b = 0; b < len; ++b) {
                        if(done[b]) continue;
                        const auto& nd = tree.decisionNode(nodeIdx[b]);
                        if(nd.isLeaf()) {
                            done[b] = true;
                        } else {
                            allDone = false;
                            auto fi   = nd.featureIndex();
                            auto th   = nd.threshold();
                            auto fval = features_(base + b, fi);
                            nodeIdx[b] = (fval < th)
                            ? nd.childNodeIndex(LEFT_NODE)
                            : nd.childNodeIndex(RIGHT_NODE);
                        }
                    }
                }

                // write out final labels
                for(size_t b = 0; b < len; ++b) {
                    labels[base - samplesBegin + b] =
                    tree.decisionNode(nodeIdx[b]).label();
                }
            }

            this->ff_send_out(
                new PWN_out_t(samplesBegin, samplesEnd, std::move(labels))
            );
            delete task; // Clean up the task after sending it out.
            return this->GO_ON; // Continue processing.
        }
        

        const andres::View<Feature>& features_;
        const std::vector<DecisionTreeType>& decisionTrees_;
    };

    struct PredCollector: ff::ff_minode_t<PWN_out_t, size_t> {
        PredCollector(
            std::vector<Label>& labels,
            andres::Marray<Probability>& probabilities
        )
            : labels_(labels),
              probabilities_(probabilities)
        {}
        size_t* svc(PWN_out_t* task) {
            auto& [samplesBegin, samplesEnd, labels] = *task;
            const auto numSamples = samplesEnd - samplesBegin;
            for(size_t ind = samplesBegin; ind < samplesEnd; ++ind) {
                labels_[ind] = labels[ind - samplesBegin];
#if defined(DEBUG)
                assert(ind < labels_.size());
#endif
                ++probabilities_(ind, labels_[ind]);
            }
            
            delete task; // Clean up the task after sending it out.
            return this->GO_ON; // Continue processing.
        }
        std::vector<Label>& labels_;
        andres::Marray<Probability>& probabilities_;
    };

    struct PredEmitter: ff::ff_monode_t<size_t,PEN_out_t>{
        PredEmitter(
            const size_t numOfPointsToPredict
        ):
            numOfPointsToPredict_(numOfPointsToPredict),
            totalChunks((numOfPointsToPredict_ + chunk_size - 1) / chunk_size)
        {}

        PEN_out_t* svc(size_t* task) {
            
            for(size_t chunk = 0; chunk < totalChunks; ++chunk) {
                this->ff_send_out(
                    new PEN_out_t(
                        chunk*chunk_size,
                        std::min((chunk+1)*chunk_size, numOfPointsToPredict_),
                        *task
                    )
                );
            }
            delete task; // Clean up the task after sending it out.
            return this->GO_ON; // Continue processing.
        }

        void eosnotify(ssize_t id) {
            // Notify the end of the stream.
            eosreceived_ = true;

        }
        const int chunk_size = CHUNK_SIZE_TO_PREDICT;
        const size_t numOfPointsToPredict_;
        const size_t totalChunks;
        
        bool eosreceived_ = false; // Flag to indicate if EOS has been received.

    };

    struct PredGenerator: ff::ff_monode_t<size_t, size_t> {

        PredGenerator(
            const std::vector<DecisionTreeType>& decisionTrees
        ):
            decisionTrees_(decisionTrees)
        {}

        size_t* svc(size_t*){

            if(currentTreeIndex_ < decisionTrees_.size()) {
                this->ff_send_out(new size_t(currentTreeIndex_));
                ++currentTreeIndex_;
                return this->GO_ON; // Continue processing.
            } else {
                return this->EOS; // No more trees to process.
            }
            
        }

        private:
            const std::vector<DecisionTreeType>& decisionTrees_;
            size_t currentTreeIndex_ = 0;
    };

    
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


template<class FEATURE, class LABEL>
inline DecisionNode<FEATURE, LABEL>&
DecisionTree<FEATURE, LABEL>::operator[](
    const size_t nodeIndex
) {
    return decisionNodes_[nodeIndex];
}

template<class FEATURE, class LABEL>
inline void DecisionTree<FEATURE, LABEL>::addEmptyNode()
{
    decisionNodes_.emplace_back();
}

template<class FEATURE, class LABEL>
inline bool DecisionTree<FEATURE, LABEL>::isEmpty() const {
    return decisionNodes_.empty();
}

/// Returns the number of nodes in the decision tree.
///
template<class FEATURE, class LABEL>
inline size_t DecisionTree<FEATURE, LABEL>::sizeTrees() const {
    return decisionNodes_.size();
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


/**
 * Learns the decision forest using this network:
 *                        ----------------------------------------
 *                   |    |       |       |    |-> WN0 -->|      |       |
 *                   |    |       |-> WNL |    |          |      |       |
 *                   |    v       |       |    |-> WN1 -->|      |       |
 *                   | -> EN0 --> |       | -->|          | --> CN0      |
 *                   |            |       |    |-> WN2 -->|              |
 *                   |            |-> WNR |    |          |              |
 *                   |            |       |    |-> WN3 -->|              |
 *                   |            <------------A2A-------->              |
 *                   |   <--------------------Farm----------------->     |
 *TrainGenerator --->|                         .                         |
 *                   |                         .                         |
 *                   |                         .                         |
 *                   |                         .                         |
 *                   |                         .                         |   
 *                   |                         .                         |       
 *                   |                         .                         |
 *                   |                         .                         |
 *                   |                         .                         |
 *  <-----------------------------Farm----------------------------------->
 * 
 * 
 * @param features A matrix in which every rows corresponds to a sample and every column corresponds to a feature.
 * @param labels A vector of labels, one for each sample.
 * @param numberOfDecisionTrees Number of decision trees to be learned.
 * @param randomSeed A random seed for the random number generator. If set to 0, a random seed will be generated automatically.
 */
template<class FEATURE, class LABEL, class PROBABILITY>
void DecisionForest<FEATURE, LABEL, PROBABILITY>::learnWithFFNetwork(
    const andres::View<Feature>& features,
    const andres::View<Label>& labels,
    const size_t& numberOfDecisionTrees,
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
    assert(numberOfDecisionTrees > 0);
    
    auto numberOfFeaturesToBeAssessed = static_cast<size_t>(
        std::ceil(std::sqrt(static_cast<double>(features.shape(1))))
    );

    using WorkerLA2A_t = typename DecisionTreeType::DecisionNodeType::WorkerLA2A;
    using WorkerRA2A_t = typename DecisionTreeType::DecisionNodeType::WorkerRA2A;

    std::vector<std::unique_ptr<TrainEmitter>> emitters;
    std::vector<std::unique_ptr<TrainCollector>> collectors;
    
    std::vector<std::unique_ptr<ff::ff_a2a>> a2a_nodes;
    std::vector<std::vector<WorkerLA2A_t*>> workersLA2A_sets;
    std::vector<std::vector<WorkerRA2A_t*>> workersRA2A_sets;
    TrainGenerator generator (numberOfDecisionTrees);
    size_t efWorkers = std::min(
        static_cast<size_t>(EN_NUM_WORKERS), 
        static_cast<size_t>(ff_numCores())
    );
    size_t wnSecWorekers = std::min(
        numberOfDecisionTrees,
        static_cast<size_t>(WN_SECOND_NUM_WORKERS)
    );
    wnSecWorekers = std::min(numberOfFeaturesToBeAssessed, wnSecWorekers);
    efWorkers = std::min(efWorkers, numberOfDecisionTrees);
    efWorkers = std::max(efWorkers, static_cast<size_t>(1)); // Ensure at least one farm is created.
    std::cout << "Using " << efWorkers << " farms for training." << std::endl;
    std::cout << "Using " << WN_FIRST_NUM_WORKERS << " workers for left set and "
              << wnSecWorekers << " workers for right set in each farm." << std::endl;

    auto farm_builder = [&](){
        std::vector<std::unique_ptr<ff::ff_farm>> farms;
        for(size_t curr_farm{0}; curr_farm < efWorkers; ++curr_farm) {
            emitters.emplace_back(std::make_unique<TrainEmitter>(
                features, labels, decisionTrees_, randomSeed
            ));

            workersLA2A_sets.emplace_back();
            for(size_t i = 0; i < WN_FIRST_NUM_WORKERS; ++i) 
                workersLA2A_sets.back().push_back(
                    new WorkerLA2A_t(
                        features, labels, numberOfFeaturesToBeAssessed,  emitters.back()->matrixSampleIndices_, randomSeed
                    )
                );

            workersRA2A_sets.emplace_back();
            for(size_t i = 0; i < wnSecWorekers; ++i) 
                workersRA2A_sets.back().push_back(
                    new WorkerRA2A_t(
                        features, labels, emitters.back()->matrixSampleIndices_
                    )
                );

            a2a_nodes.push_back(std::make_unique<ff::ff_a2a>());
            a2a_nodes.back()->add_firstset(workersLA2A_sets.back());
            a2a_nodes.back()->add_secondset(workersRA2A_sets.back());
            a2a_nodes.back()->no_mapping();

           

            collectors.emplace_back(std::make_unique<TrainCollector>(
                features, labels, numberOfFeaturesToBeAssessed, emitters.back()->matrixSampleIndices_
            ));
            farms.emplace_back(std::make_unique<ff::ff_farm>(false));
            farms[curr_farm]->add_emitter(emitters.back().get());
            farms[curr_farm]->add_workers({a2a_nodes.back().get()});
            farms[curr_farm]->add_collector(reinterpret_cast<ff::ff_node*>(collectors.back().get()));
            farms[curr_farm]->blocking_mode();
            farms[curr_farm]->wrap_around();
            
        }
        return std::move(farms);
    };
    
    decisionTrees_.resize(numberOfDecisionTrees);
    auto inner_farms = farm_builder();
    std::vector<std::unique_ptr<ff::ff_node>> farm_nodes;
    for(auto& farm : inner_farms) {
        farm_nodes.push_back(std::move(farm));
    }

    ff::ff_Farm<FEN_in_t> generalFarm(
        std::move(farm_nodes),
        generator
    );  
    
    generalFarm.remove_collector();
    generalFarm.blocking_mode();


    if(generalFarm.run_and_wait_end() < 0) {
        throw std::runtime_error("Error during decision forest learning with FastFlow network.");
    }
    // Clean up workers
    for(auto& worker_set : workersLA2A_sets)
        for(auto& worker : worker_set)
            delete worker;

    for(auto& worker_set : workersRA2A_sets)
        for(auto& worker : worker_set)
            delete worker;

    
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

    std::vector<Label> labels(features.shape(0));

    PredGenerator generator(decisionTrees_);
    std::vector<std::unique_ptr<PredEmitter>> emitters;
    std::vector<std::unique_ptr<PredCollector>> collectors;
    
    std::fill(labelProbabilities.begin(), labelProbabilities.end(), Probability());


    auto numWorkers = std::min(static_cast<size_t>(MAX_PWN_NUM_WORKERS), features.shape(0) / CHUNK_SIZE_TO_PREDICT);
    if(numWorkers == 0) {
        numWorkers = 1; // Ensure at least one worker is created.
    }
    auto cpu_cores = static_cast<size_t>(ff_numCores());
    size_t pfInPar = std::min(std::min(static_cast<size_t>(PEN_NUM_WORKERS), decisionTrees_.size()), cpu_cores);
    std::cout << "Using " << pfInPar << " parallel farms for prediction." << std::endl;
    std::cout << "Using " << numWorkers << " workers per farm." << std::endl;
    std::vector<std::vector<std::unique_ptr<PredWorker>>> workers(pfInPar);

    auto farm_builder = [&](){
        std::vector<std::unique_ptr<ff::ff_farm>> farms;
        for(size_t curr_farm{0}; curr_farm < pfInPar; ++curr_farm) {
            emitters.emplace_back(std::make_unique<PredEmitter>(features.shape(0)));
            collectors.emplace_back(std::make_unique<PredCollector>(
                labels,
                labelProbabilities
            ));
            
            farms.emplace_back(std::make_unique<ff::ff_farm>(false));
            farms[curr_farm]->add_emitter(emitters.back().get());
            for (size_t i = 0; i < numWorkers; ++i) {
                workers[curr_farm].emplace_back(
                    std::make_unique<PredWorker>(features, decisionTrees_)
                );
                farms[curr_farm]->add_workers({workers[curr_farm].back().get()});
            }
            farms[curr_farm]->add_collector(collectors.back().get());
            farms[curr_farm]->blocking_mode();
        }
        return std::move(farms);
    };

    auto inner_farms = farm_builder();
    std::vector<std::unique_ptr<ff::ff_node>> farm_nodes;
    for(auto& farm : inner_farms) {
        farm_nodes.push_back(std::move(farm));
    }
    ff::ff_Farm<PEN_out_t> generalFarm(
        std::move(farm_nodes),
        generator
    );
    generalFarm.remove_collector();
    generalFarm.set_scheduling_ondemand();
    generalFarm.blocking_mode();
    if(generalFarm.run_and_wait_end() < 0) {
        throw std::runtime_error("Error during decision forest prediction with FastFlow network.");
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

#endif //