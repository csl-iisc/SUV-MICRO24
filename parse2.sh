#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

footprints=(8192 3500 2610 4096 5120 8192 6912 4096 6912 3072 6912 5760 4096 3884)

oversub=(50)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

filename="fig6.csv"
echo -n "" > $filename
echo -n "," >> $filename
for os in ${oversub[@]}; do
    echo -n "pagefaults" >> $filename
    echo -n "$os.uvm" >> $filename
    echo -n "," >> $filename
    echo -n "$os.suv" >> $filename
    echo -n "," >> $filename
done

for ((idx=0; idx<${#benchmarks[@]}; ++idx)); do
    benchmark=${benchmarks[idx]}
    footprint=${footprints[idx]} 
    echo "Processing $benchmark $footprint"
    echo "" >> $filename
    echo -n $benchmark >> $filename
    echo -n "," >> $filename
    for os in ${oversub[@]}; do
        cd ${pwd0}
        cd eval
        cd $benchmark
        echo $benchmark $os
        uvmpf=$(tail -n 3 uvm.${os}.pf.txt | grep "page fault" | awk {'print $7}')
        echo "$uvmpf"
        suvpf=$(tail -n 3 suv.${os}.pf.txt | grep "page fault" | awk {'print $7}')
        echo "$suvpf"
        cd ${pwd0}
        echo -n $uvmpf >> $filename
        echo -n "," >> ${filename}
        echo -n "$suvpf" >> $filename
        echo -n "," >> $filename
        echo ""
    done
done

