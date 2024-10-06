#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

oversub=(0)
cp penguin-suv.h penguin.h
for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    echo "Processing $benchmark 0"
    cd ${pwd0}
    cd eval
    cd $benchmark
    echo $(pwd)
    echo "uvm.0.out"
    rm uvm.0.out
    sed -i "s/#define MiB.*/#define MiB 0/g" main.cu
    bash compile_app.sh $SUVHOME $SUVHOME/llvm/ uvm.0.out
    cd ${pwd0}
    echo ""
done
