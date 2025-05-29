# Distributed Parallel Random Forest

**SPM Course Project 2 - A.A. 24/25**

## Project Overview

This project implements a scalable, parallel, and distributed version of the Random Forest algorithm for classification tasks. The implementation handles "large" datasets stored in CSV format containing numerical features and binary labels, with input from the UCI repository and results written to output files.

## Project Description

Design and implement three versions of the Random Forest algorithm:
1. **Single-node shared-memory** versions using FastFlow and OpenMP
2. **Multi-node hybrid** version combining MPI with either FastFlow or OpenMP
3. **Performance evaluation** across various parameters and configurations

### Datasets

The implementation uses datasets from the UCI Machine Learning Repository:
- [SUSY](https://archive.ics.uci.edu/ml/datasets/SUSY)
- [Covertype](https://archive.ics.uci.edu/ml/datasets/covertype)
- [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- Other suitable classification datasets

**Repository Link:** https://archive.ics.uci.edu/ml/index.php

## Implementation Tasks

### 1. Single-node Versions (Shared-memory)

#### FastFlow Implementation
- Parallel Random Forest using FastFlow framework
- Optimized for shared-memory multicore systems
- Maintains consistency with OpenMP version results

#### OpenMP Implementation
- Parallel Random Forest using OpenMP pragmas
- Thread-based parallelization
- Same classification results as FastFlow version

**Validation Requirements:**
- Compare against scikit-learn implementation
- Use identical train/test splits
- Report accuracy and macro-averaged F1-score metrics
- Test on small datasets (Iris or reduced SUSY subset)

### 2. Multi-node Hybrid Version

#### Distributed Implementation
- Combines MPI with either FastFlow or OpenMP
- Parallelizes both training and prediction phases
- **Training Phase:** Distributed decision tree construction
- **Prediction Phase:** Distributed model inference

#### Architecture Decisions
- Clear justification for FastFlow vs OpenMP choice
- Optimal load balancing strategies
- Efficient inter-node communication patterns

### 3. Performance Evaluation

#### Parameter Variations
The evaluation systematically varies:
- **a)** Number of decision trees in the forest
- **b)** Dataset size and characteristics (samples × features)
- **c)** Number of FastFlow/OpenMP threads per node

#### Scalability Analysis
- **Single-node:** Speedup and efficiency curves
- **Multi-node:** Strong and weak scalability (up to 8 nodes)
- **Phase-specific:** Separate analysis for training and prediction
- **Hybrid analysis:** MPI processes vs threads per process

#### Cluster Configuration
- Evaluation on SPM cluster
- Up to 8 compute nodes
- Detailed MPI/thread configuration analysis

### 4. Performance Analysis

#### Cost Model Development
- Approximate cost model for distributed Random Forest
- Theoretical complexity analysis
- Performance bottleneck identification

#### Implementation Challenges
- Detailed description of encountered challenges
- Solutions and optimizations applied
- Resource utilization maximization strategies

#### Optimization Goals
- Minimize parallelization overhead
- Maximize resource utilization
- Efficient memory management
- Optimal communication patterns

## Project Structure

```
project/
├── src/                    # Source code files
│   ├── fastflow/          # FastFlow implementation
│   ├── openmp/            # OpenMP implementation
│   ├── mpi/               # MPI hybrid implementation
│   └── common/            # Shared utilities
├── scripts/               # Compilation and execution scripts
├── data/                  # Dataset files
├── results/               # Output and performance results
├── docs/                  # Documentation
└── README.md             # This file
```

## Compilation and Execution

### Prerequisites
- C++ compiler with OpenMP support
- FastFlow framework
- MPI implementation (OpenMPI/MPICH)
- Access to SPM cluster

### Build Scripts
```bash
# Compile all versions
./scripts/compile_all.sh

# Single-node versions
./scripts/compile_fastflow.sh
./scripts/compile_openmp.sh

# Distributed version
./scripts/compile_mpi.sh
```

### Execution Scripts
```bash
# Single-node execution
./scripts/run_single_node.sh <dataset> <trees> <threads>

# Multi-node execution
./scripts/run_distributed.sh <dataset> <trees> <nodes> <threads_per_node>

# Performance benchmarks
./scripts/benchmark_all.sh
```

## Performance Metrics

### Evaluation Criteria
- **Accuracy:** Classification correctness
- **F1-Score:** Macro-averaged F1-score
- **Speedup:** Parallel efficiency measurement
- **Scalability:** Strong and weak scaling analysis
- **Resource Utilization:** CPU and memory efficiency

### Benchmarking Framework
- Automated performance testing
- Statistical significance validation
- Comprehensive result logging
- Visualization of performance trends

## Deliverables

### Source Code
- Complete implementation of all three versions
- Well-documented and modular code structure
- Compilation and execution scripts
- Performance benchmarking tools

### Documentation
- **PDF Report:** Maximum 15 pages
  - Implementation details
  - Performance analysis
  - Challenges and solutions
  - Optimization strategies
- **Code Documentation:** Inline comments and API docs

### Submission Requirements
- **Format:** Single ZIP file: `SPM_project2_<YourName>.zip`
- **Email:** massimo.torquati@unipi.it
- **Subject:** "SPM Project"
- **Contents:** Source code + PDF report

## Development Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1 | Week 1-2 | Sequential baseline + Single-node implementations |
| 2 | Week 3-4 | Multi-node distributed version |
| 3 | Week 5-6 | Performance evaluation and optimization |
| 4 | Week 7 | Documentation and final submission |

## Key Features

### Algorithm Correctness
- Validated against scikit-learn
- Consistent results across implementations
- Comprehensive testing framework

### Performance Optimization
- Minimized parallelization overhead
- Optimized memory access patterns
- Efficient load balancing
- Smart data partitioning

### Scalability Design
- Linear scalability targets
- Efficient resource utilization
- Minimal communication overhead
- Adaptive load distribution

## Getting Started

1. **Clone the repository**
2. **Install dependencies** (FastFlow, MPI, OpenMP)
3. **Download datasets** from UCI repository
4. **Compile implementations** using provided scripts
5. **Run validation tests** on small datasets
6. **Execute performance benchmarks**
7. **Analyze results** and generate reports

## Contact Information

For questions and support:
- **Course:** SPM (Parallel and Distributed Systems)
- **Academic Year:** 2024/2025
- **Instructor:** Prof. Massimo Torquati
- **Email:** massimo.torquati@unipi.it

---

*This project demonstrates advanced concepts in parallel and distributed computing, focusing on real-world machine learning applications and performance optimization techniques.*

