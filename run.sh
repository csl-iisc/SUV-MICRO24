#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

footprints=(8192 3500 2610 4096 5120 8192 6912 4096 6912 3072 6912 5760 4096 3884)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

oversub=(15 30 50)

for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    footprint=${footprints[idx]} 
    echo "Processing $benchmark $footprint"
    for os in ${oversub[@]}; do
        cd ${pwd0}
        bash driver_change.sh 0 64k 256
        cd eval
        cd $benchmark
        echo $(pwd)
        echo "uvm.${os}.out"
        echo "running"
        ls -ltr uvm.${os}.out
        ./uvm.${os}.out &> uvm.${os}.txt
        sudo dmesg | tail &> uvm.${os}.pf.txt
        sleep 3 
        echo "running"
        ls -ltr suv.${os}.out
        ./suv.${os}.out &> suv.${os}.txt
        sudo dmesg | tail &> suv.${os}.pf.txt
        sleep 3 
        cd ${pwd0}
        bash driver_change.sh 1 64k 256
        cd eval
        cd $benchmark
        ls -ltr uvm.${os}.out
        ./uvm.${os}.out &> ac.${os}.txt
        sudo dmesg | tail &> ac.${os}.pf.txt
        sleep 3 
        cd ${pwd0}
        echo ""
    done
done

cd ${pwd0}
echo ""

oversub=(50)
bash driver_change.sh 0 64k 256

benchmarks=("2dconv" "alexnet"  "bicg"  "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt")
footprints=(8192 3500  4096  8192 6912 4096 6912 3072 6912 5760 4096)
for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    footprint=${footprints[idx]} 
    echo "Processing $benchmark $footprint"
    for os in ${oversub[@]}; do
        cd ${pwd0}
        bash driver_change.sh 0 64k 256
        cd eval
        cd $benchmark
        echo $(pwd)
        echo "running"
        ls -ltr sc.${os}.out
        ./sc.${os}.out &> sc.${os}.txt
        sudo dmesg | tail &> sc.${os}.pf.txt
        sleep 3 
        cd ${pwd0}
        echo ""
    done
done

cd ${pwd0}
echo ""

# cd ${pwd0}
# echo ${pwd0}
# oversub=(0)
# for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
#     benchmark=${benchmarks[idx]}
#     footprint=${footprints[idx]} 
#     echo "Processing $benchmark $footprint"
#     for os in ${oversub[@]}; do
#         cd ${pwd0}
#         bash driver_change.sh 0 64k 256
#         cd eval
#         cd $benchmark
#         echo $(pwd)
#         echo "uvm.${os}.out"
#         echo "running"
#         ls -ltr uvm.${os}.out
#         ./uvm.${os}.out &> uvm.${os}.txt
#         cd ${pwd0}
#         echo ""
#     done
# done

