#! /bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <target to execute> [extra args]"
  echo "Available targets: ff_impl_exp1: compiles and runs the first fastflow implementation"
  echo "                   ff_impl_exp2: compiles and the second fastflow implementation"
  echo "                   mpi_impl: compiles and runs the MPI implementation (uses mpirun with MPI_PROCS or optional second arg)"
  echo "                   omp_impl: compiles and runs the OpenMP implementation"
  echo "                   run_ff_impl_comparison: compiles and runs both fastflow implementations for comparison"
  echo "                   run_sequential: compiles and runs the sequential implementation"
  echo "                   run_sequential_validations: compiles the sequential random forest implementation for validation"

  exit 1
fi

SRUN_ARGS="--mpi=pmix --ntasks-per-node=1 --time=00:10:00"

cd build || exit 1

rm -rf ./*

# switch to an array
declare -a CMAKE_ARGS
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)

if [[ "$1" == "sequential_impl" ]]; then
  echo "Building with sequential implementation enabled."
  CMAKE_ARGS+=(-DBUILD_SEQUENTIAL=ON)
fi

if [[ "$1" == "run_sequential_validation" ]]; then
  echo "Building with sequential validation enabled."
  CMAKE_ARGS+=(-DBUILD_VALSEQ=ON)
fi

curr_dir_name=$(basename "$PWD")
if [[ "$curr_dir_name" != "build" ]]; then
  echo "Error: This script must be run from the 'build' directory."
  exit 1
fi

rm -rf ./*
echo "CMAKE_ARGS: ${CMAKE_ARGS[*]}"

cmake "${CMAKE_ARGS[@]}" ../
cmake --build . --target "$1" -j 4

cd ..

TRAIN_DATA=./data/susy/train_susy.csv
TEST_DATA=./data/susy/test_susy.csv

if [[ "$1" == "mpi_impl" ]]; then
  MPI_PROCS="${MPI_PROCS:-${2:-2}}"
  if command -v srun >/dev/null 2>&1; then
    echo "Running within SLURM environment using srun with $MPI_PROCS processes."
    srun -n "$MPI_PROCS" $SRUN_ARGS ./build/bin/mpi_impl "$TRAIN_DATA" "$TEST_DATA"
    exit 0
  fi
    
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "Error: mpirun command not found. Please ensure an MPI runtime is installed and available in PATH."
    exit 1
  fi
  if ! [[ "$MPI_PROCS" =~ ^[0-9]+$ ]] || [ "$MPI_PROCS" -lt 2 ]; then
    echo "Error: MPI processes count must be an integer greater than or equal to 2."
    exit 1
  fi
  echo "Running MPI implementation with $MPI_PROCS processes."
  mpirun -np "$MPI_PROCS" ./build/bin/mpi_impl "$TRAIN_DATA" "$TEST_DATA"
elif [[ ! "$1" =~ ^run ]]; then
  ./build/bin/"$1" "$TRAIN_DATA" "$TEST_DATA"
fi
