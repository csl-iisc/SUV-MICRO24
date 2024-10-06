#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

footprints=(8192 3500 2610 4096 5120 8192 6912 4096 6912 3072 6912 5760 4096 3884)

oversub=(15 30 50)

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
        echo "uvm.${os}.out"
        rm uvm.${os}.out
        echo $(pwd)
        bash compile_app.sh $SUVHOME $SUVHOME/llvm/ uvm.${os}.out
        echo "suv.${os}.out"
        rm suv.${os}.out
        bash run_passes.sh  $SUVHOME $SUVHOME/llvm/ suv.${os}.out
        cd ${pwd0}
        echo ""
    done
done

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
        echo "sc.${os}.out"
        rm sc.${os}.out
        bash run_sc.sh $SUVHOME $SUVHOME/llvm/ sc.${os}.out
        cd ${pwd0}
        echo ""
    done
done

# cd ${pwd0}
# echo ${pwd0}
# oversub=(0)
# cp penguin-suv.h penguin.h
# for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
#     benchmark=${benchmarks[idx]}
#     footprint=${footprints[idx]} 
#     echo "Processing $benchmark $footprint"
#     for os in ${oversub[@]}; do
#         bash set_mem_reserve.sh $benchmark $footprint $os
#         cd ${pwd0}
#         cd eval
#         cd $benchmark
#         echo $(pwd)
#         echo "uvm.${os}.out"
#         rm uvm.${os}.out
#         bash compile_app.sh $SUVHOME $SUVHOME/llvm/ uvm.${os}.out
#         cd ${pwd0}
#         echo ""
#     done
# done

