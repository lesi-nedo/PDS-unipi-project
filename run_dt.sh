#! /bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <itarget to execute>"
  exit 1
fi

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
if [[ ! "$1" =~ ^run ]]; then
  ./build/bin/"$1" ./data/susy/train_susy.csv ./data/susy/test_susy.csv
fi
