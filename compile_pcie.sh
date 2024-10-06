#!/bin/bash

tx=$1

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

footprints=(8192 3500 2610 4096 5120 8192 6912 4096 6912 3072 6912 5760 4096 3884)

oversub=(50)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

cp penguin-suv.h penguin.h
for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    footprint=${footprints[idx]} 
    echo "Processing $benchmark $footprint"
    for os in ${oversub[@]}; do
        bash set_mem_reserve.sh $benchmark $footprint $os
        cd ${pwd0}
        cd eval
        cd $benchmark
        echo $(pwd)
        echo "uvm.${os}.${tx}.out"
        rm uvm.${os}.${tx}.out
        bash compile_app.sh $SUVHOME $SUVHOME/llvm/ uvm.${os}.${tx}.out
        echo "suv.${os}.${tx}.out"
        rm suv.${os}.${tx}.out
        bash run_passes.sh $SUVHOME $SUVHOME/llvm/ suv.${os}.${tx}.out
        cd ${pwd0}
        echo ""
    done
done

# redefine for SC
benchmarks=("2dconv" "alexnet"  "bicg"  "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt")
footprints=(8192 3500  4096  8192 6912 4096 6912 3072 6912 5760 4096)

oversub=(50)

cd ${pwd0}
echo ""

cp penguin-sc.h penguin.h

for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    footprint=${footprints[idx]} 
    echo "Processing $benchmark $footprint"
    for os in ${oversub[@]}; do
        bash set_mem_reserve.sh $benchmark $footprint $os
        cd ${pwd0}
        cd eval
        cd $benchmark
        echo $(pwd)
        echo "sc.${os}.${tx}.out"
        rm sc.${os}.${tx}.out
        bash run_sc.sh $SUVHOME $SUVHOME/llvm/ sc.${os}.${tx}.out
        cd ${pwd0}
        echo ""
    done
done

