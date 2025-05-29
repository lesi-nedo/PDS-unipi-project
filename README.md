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
├── CMakeLists.txt         # Main CMake configuration
├── cmake/                 # CMake modules and find scripts
│   ├── FindFastFlow.cmake # FastFlow detection
│   └── modules/           # Additional CMake modules
├── src/                   # Source code files
│   ├── fastflow/          # FastFlow implementation
│   │   ├── CMakeLists.txt # FastFlow target configuration
│   │   └── *.cpp/*.hpp    # Source and header files
│   ├── openmp/            # OpenMP implementation
│   │   ├── CMakeLists.txt # OpenMP target configuration
│   │   └── *.cpp/*.hpp    # Source and header files
│   ├── mpi/               # MPI hybrid implementation
│   │   ├── CMakeLists.txt # MPI target configuration
│   │   └── *.cpp/*.hpp    # Source and header files
│   └── utils/             # Shared utilities and common code
│       ├── CMakeLists.txt # Utility library configuration
│       └── *.cpp/*.hpp    # Utility functions
├── build/                 # Build directory (generated)
├── bin/                   # Compiled executables (generated)
├── scripts/               # Utility scripts (benchmarking, data processing)
├── data/                  # Dataset files
├── results/               # Output and performance results
├── report/                # Documentation and analysis
└── README.md              # This file
```

## Compilation and Execution

### Prerequisites
- C++ compiler with C++17 support (g++ 7.0+ or clang++ 5.0+)
- CMake 3.12 or higher
- FastFlow framework
- MPI implementation (OpenMPI/MPICH)
- OpenMP support
- Access to SPM cluster

### Build System
The project uses CMake for cross-platform build configuration and dependency management.

#### Quick Start
```bash
# Create build directory
mkdir build && cd build

# Configure the project
cmake ..

# Build all targets
cmake --build . --parallel 4

# Or use make (if using Unix Makefiles generator)
make -j4
```

#### Configuration Options
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Specify custom FastFlow path
cmake -DFastFlow_ROOT=/path/to/fastflow ..

# Enable specific components only
cmake -DBUILD_FASTFLOW=ON -DBUILD_OPENMP=ON -DBUILD_MPI=OFF ..

# Cross-compilation for cluster
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/cluster-toolchain.cmake ..
```

#### Build Targets
```bash
# Build specific implementations
cmake --build . --target rf_fastflow
cmake --build . --target rf_openmp
cmake --build . --target rf_mpi

# Build utilities library
cmake --build . --target rf_utils

# Run tests
cmake --build . --target test
# or
ctest

# Install executables
cmake --build . --target install
```

### Execution
```bash
# Single-node FastFlow execution
./bin/rf_fastflow <dataset> <trees> <threads>

# Single-node OpenMP execution
./bin/rf_openmp <dataset> <trees> <threads>

# Multi-node MPI execution
mpirun -np <processes> -hostfile hostfile ./bin/rf_mpi <dataset> <trees> <threads_per_process>

# Performance benchmarks
./scripts/benchmark.sh
```

### Example Usage
```bash
# Configure and build everything
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel 4

# Run on Iris dataset with 100 trees, 4 threads
./bin/rf_fastflow ../data/iris.csv 100 4
./bin/rf_openmp ../data/iris.csv 100 4

# Distributed execution on 4 nodes, 2 processes per node
mpirun -np 8 -npernode 2 ./bin/rf_mpi ../data/susy.csv 500 2

# Run comprehensive benchmarks
cd .. && ./scripts/benchmark.sh
```

### CMake Features for This Project

#### Automatic Dependency Detection
- **FindOpenMP.cmake** - Automatically detects OpenMP support
- **FindMPI.cmake** - Locates MPI implementation
- **FindFastFlow.cmake** - Custom module to find FastFlow installation

#### Cross-Platform Support
- Works on Linux, macOS, and Windows
- Supports different compilers (GCC, Clang, Intel)
- Handles different MPI implementations automatically

#### Advanced Configuration
```bash
# Generate compile_commands.json for IDEs
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# Enable AddressSanitizer for debugging
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON ..

# Profile-guided optimization
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_PGO=ON ..

# Generate documentation
cmake --build . --target docs
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

---

*This project demonstrates advanced concepts in parallel and distributed computing, focusing on real-world machine learning applications and performance optimization techniques.*

