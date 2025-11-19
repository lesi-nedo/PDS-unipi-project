#!/bin/bash
# Complete evaluation workflow script

set -e

echo "=========================================="
echo "Complete Performance Evaluation Workflow"
echo "=========================================="

PROJECT_ROOT="/home/lesi-nedo/Desktop/master/first-year/second-semester/PDS/project"
cd "$PROJECT_ROOT"

# Function to run evaluation
run_evaluation() {
    local target=$1
    local impl_name=$2
    local results_path=$3
    local train_data=$4
    local test_data=$5
    
    echo ""
    echo "=========================================="
    echo "Running: $impl_name"
    echo "=========================================="
    
    cd "$PROJECT_ROOT/build"
    
    echo "Building $target..."
    cmake --build . --target "$target"
    
    echo ""
    echo "Executing evaluation..."
    ./bin/"$target" "$train_data" "$test_data"
    
    echo ""
    echo "Checking results..."
    if [ -f "$results_path/comprehensive_performance.csv" ]; then
        local rows=$(wc -l < "$results_path/comprehensive_performance.csv")
        echo "✓ Generated comprehensive_performance.csv with $rows rows"
    else
        echo "✗ comprehensive_performance.csv not found"
        return 1
    fi
    
    if [ -f "$results_path/cost_model_validation.txt" ]; then
        echo "✓ Generated cost_model_validation.txt"
    else
        echo "✗ cost_model_validation.txt not found"
        return 1
    fi
}

# Function to generate analysis
generate_analysis() {
    local results_path=$1
    local impl_name=$2
    
    echo ""
    echo "=========================================="
    echo "Generating Analysis: $impl_name"
    echo "=========================================="
    
    cd "$PROJECT_ROOT/analysis"
    
    if [ ! -f "$results_path/comprehensive_performance.csv" ]; then
        echo "✗ No comprehensive_performance.csv found in $results_path"
        return 1
    fi
    
    echo "Running analysis script..."
    python3 plot_comprehensive_results.py "$results_path/comprehensive_performance.csv" \
        --output-dir "$results_path/analysis"
    
    echo ""
    echo "Generated files:"
    ls -lh "$results_path/analysis/"
}

# Parse command line arguments
IMPL=${1:-all}
DATASET=${2:-iris}

# Set dataset paths
case $DATASET in
    iris)
        TRAIN_DATA="$PROJECT_ROOT/data/iris/train_iris.csv"
        TEST_DATA="$PROJECT_ROOT/data/iris/test_iris.csv"
        echo "Using IRIS dataset (small, for testing)"
        ;;
    susy)
        TRAIN_DATA="$PROJECT_ROOT/data/susy/train_susy.csv"
        TEST_DATA="$PROJECT_ROOT/data/susy/test_susy.csv"
        echo "Using SUSY dataset (large, for full evaluation)"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Usage: $0 [all|ff1|ff2|omp|mpi] [iris|susy]"
        exit 1
        ;;
esac

# Check if data files exist
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$TEST_DATA" ]; then
    echo "Error: Dataset files not found"
    echo "  Train: $TRAIN_DATA"
    echo "  Test: $TEST_DATA"
    exit 1
fi

# Build directory setup
mkdir -p "$PROJECT_ROOT/build"
cd "$PROJECT_ROOT/build"

echo ""
echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python3) ..

# Run evaluations based on implementation selection
case $IMPL in
    ff1)
        run_evaluation "ff_impl_exp1" "FastFlow Experiment 1" \
            "$PROJECT_ROOT/results/fastflow_impl1" "$TRAIN_DATA" "$TEST_DATA"
        generate_analysis "$PROJECT_ROOT/results/fastflow_impl1" "FastFlow Exp1"
        ;;
    ff2)
        run_evaluation "ff_impl_exp2" "FastFlow Experiment 2" \
            "$PROJECT_ROOT/results/fastflow_impl2" "$TRAIN_DATA" "$TEST_DATA"
        generate_analysis "$PROJECT_ROOT/results/fastflow_impl2" "FastFlow Exp2"
        ;;
    omp)
        echo ""
        echo "Reconfiguring for OpenMP..."
        cd "$PROJECT_ROOT/build"
        rm -rf *
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_OPENMP=ON -DPython3_EXECUTABLE=$(which python3) ..
        
        run_evaluation "omp_impl" "OpenMP Implementation" \
            "$PROJECT_ROOT/results/openmp" "$TRAIN_DATA" "$TEST_DATA"
        generate_analysis "$PROJECT_ROOT/results/openmp" "OpenMP"
        ;;
    mpi)
        echo ""
        echo "Running MPI Implementation..."
        echo "Note: MPI requires at least 2 processes. Adjust mpirun parameters as needed."
        cd "$PROJECT_ROOT/build"
        
        # Check if mpi_impl exists, if not build it
        if [ ! -f "bin/mpi_impl" ]; then
            echo "Building mpi_impl..."
            cmake --build . --target mpi_impl
        fi
        
        echo ""
        echo "Executing MPI evaluation with 8 processes..."
        mpirun -np 8 ./bin/mpi_impl "$TRAIN_DATA" "$TEST_DATA"
        
        echo ""
        echo "Checking MPI results..."
        if [ -f "$PROJECT_ROOT/results/mpi/mpi_comprehensive_performance.csv" ]; then
            local rows=$(wc -l < "$PROJECT_ROOT/results/mpi/mpi_comprehensive_performance.csv")
            echo "✓ Generated mpi_comprehensive_performance.csv with $rows rows"
        fi
        
        # Generate MPI-specific analysis
        echo ""
        echo "Generating MPI Analysis..."
        cd "$PROJECT_ROOT/analysis"
        python3 plot_mpi_results.py "$PROJECT_ROOT/results/mpi/mpi_comprehensive_performance.csv"
        ;;
    all)
        echo ""
        echo "Running all implementations..."
        echo ""
        
        # FastFlow Exp1
        run_evaluation "ff_impl_exp1" "FastFlow Experiment 1" \
            "$PROJECT_ROOT/results/fastflow_impl1" "$TRAIN_DATA" "$TEST_DATA"
        
        # FastFlow Exp2
        run_evaluation "ff_impl_exp2" "FastFlow Experiment 2" \
            "$PROJECT_ROOT/results/fastflow_impl2" "$TRAIN_DATA" "$TEST_DATA"
        
        # OpenMP
        echo ""
        echo "Reconfiguring for OpenMP..."
        cd "$PROJECT_ROOT/build"
        rm -rf *
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_OPENMP=ON -DPython3_EXECUTABLE=$(which python3) ..
        
        run_evaluation "omp_impl" "OpenMP Implementation" \
            "$PROJECT_ROOT/results/openmp" "$TRAIN_DATA" "$TEST_DATA"
        
        # MPI (if available)
        echo ""
        echo "Checking for MPI..."
        if command -v mpirun &> /dev/null; then
            echo "MPI found. Running MPI evaluation..."
            cd "$PROJECT_ROOT/build"
            if [ -f "bin/mpi_impl" ]; then
                mpirun -np 8 ./bin/mpi_impl "$TRAIN_DATA" "$TEST_DATA"
                cd "$PROJECT_ROOT/analysis"
                python3 plot_mpi_results.py "$PROJECT_ROOT/results/mpi/mpi_comprehensive_performance.csv"
            else
                echo "⚠ mpi_impl binary not found, skipping MPI evaluation"
            fi
        else
            echo "⚠ MPI not found, skipping MPI evaluation"
        fi
        
        # Generate all analyses
        echo ""
        echo "=========================================="
        echo "Generating All Analyses"
        echo "=========================================="
        
        generate_analysis "$PROJECT_ROOT/results/fastflow_impl1" "FastFlow Exp1"
        generate_analysis "$PROJECT_ROOT/results/fastflow_impl2" "FastFlow Exp2"
        generate_analysis "$PROJECT_ROOT/results/openmp" "OpenMP"
        ;;
    *)
        echo "Unknown implementation: $IMPL"
        echo "Usage: $0 [all|ff1|ff2|omp|mpi] [iris|susy]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✓ Evaluation Workflow Complete!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  - FastFlow Exp1: results/fastflow_impl1/"
echo "  - FastFlow Exp2: results/fastflow_impl2/"
echo "  - OpenMP:        results/openmp/"
echo ""
echo "Each directory contains:"
echo "  - comprehensive_performance.csv (raw data)"
echo "  - cost_model_validation.txt (model analysis)"
echo "  - analysis/ (plots and summary report)"
echo ""
echo "View the summary reports:"
echo "  cat results/*/analysis/performance_summary.txt"
echo ""
