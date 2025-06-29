#! /bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <itarget to execute>"
  exit 1
fi

cd build || exit 1

CMAKE_ARGS=""

if [[ "$1" == "sequential_impl" ]]; then
  CMAKE_ARGS+="-DBUILD_SEQUENTIAL=ON"
fi

curr_dir_name=$(basename "$PWD")

if [[ "$curr_dir_name" != "build" ]]; then
  echo "Error: This script must be run from the 'build' directory."
  exit 1
fi

rm -rf ./*

cmake $CMAKE_ARGS ../

cmake --build . --target "$1" -- -j 4

cd ..

./build/bin/"$1" ./data/susy/train_susy.csv ./data/susy/test_susy.csv
