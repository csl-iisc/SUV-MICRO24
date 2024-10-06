#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

oversub=(0)
for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    echo "Processing $benchmark"
    cd ${pwd0}
    bash driver_change.sh 0 64k 256
    cd eval
    cd $benchmark
    echo $(pwd)
    echo "uvm.0.out"
    echo "running"
    ls -ltr uvm.0.out
    ./uvm.0.out &> uvm.0.txt
    cd ${pwd0}
    echo ""
done
