#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

oversub=(50)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

filename="fig7.csv"
echo -n "exectime" > $filename
echo -n "," >> $filename
for os in ${oversub[@]}; do
    echo -n "$os.suv" >> $filename
    echo -n "," >> $filename
    echo -n "$os.ac" >> $filename
    echo -n "," >> $filename
    echo -n "$os.sc" >> $filename
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
        suvtime=$(grep "GPU.Parser.Time" suv.${os}.txt | awk {'print $2}')
        echo "$suvtime"
        actime=$(grep "GPU.Parser.Time" ac.${os}.txt | awk {'print $2}')
        echo "$actime"
        if [ $benchmark == "xsbench" ] || [ $benchmark == "bfs" ] || [ $benchmark == "bptree" ] ; then
            sctime=$(grep "GPU.Parser.Time" uvm.${os}.txt | awk {'print $2}')
        else
            sctime=$(grep "GPU.Parser.Time" sc.${os}.txt | awk {'print $2}')
        fi
        echo "$sctime"
        cd ${pwd0}
        echo -n $suvtime >> $filename
        echo -n "," >> $filename
        echo -n "$actime" >> $filename
        echo -n "," >> ${filename}
        echo -n "$sctime" >> $filename
        echo -n "," >> ${filename}
        echo ""
    done
done

