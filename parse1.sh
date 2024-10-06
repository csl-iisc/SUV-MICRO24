#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

footprints=(8192 3500 2610 4096 5120 8192 6912 4096 6912 3072 6912 5760 4096 3884)

oversub=(15 30 50)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

filename="fig5.csv"
echo -n "exectime" > $filename
echo -n "," >> $filename
for os in ${oversub[@]}; do
    echo -n "0.uvm" >> $filename
    echo -n "," >> $filename
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
    cd ${pwd0}
    cd eval
    cd $benchmark
    echo $benchmark 0
    uvmtime=$(grep "GPU.Parser.Time" uvm.0.txt | awk {'print $2}')
    echo "$uvmtime"
    cd ${pwd0}
    echo -n "$uvmtime" >> $filename
    echo -n "," >> ${filename}
    for os in ${oversub[@]}; do
        cd ${pwd0}
        cd eval
        cd $benchmark
        echo $benchmark $os
        uvmtime=$(grep "GPU.Parser.Time" uvm.${os}.txt | awk {'print $2}')
        echo "$uvmtime"
        suvtime=$(grep "GPU.Parser.Time" suv.${os}.txt | awk {'print $2}')
        echo "$suvtime"
        cd ${pwd0}
        echo -n "$uvmtime" >> $filename
        echo -n "," >> ${filename}
        echo -n $suvtime >> $filename
        echo -n "," >> $filename
        echo ""
    done
done

