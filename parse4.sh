#!/bin/bash

benchmarks=("2dconv" "alexnet" "bfs" "bicg" "bptree" "doitgen" "fdtd" "fw" "gemm" "gramschmit" "hellinger-cuda" "mm" "mvt" "xsbench")

oversub=(50)

pwd0=$(pwd) # the root folder of the artifact
echo ${pwd0}
echo ""

filename="fig8.csv"
echo -n "tx" > $filename
echo -n "," >> $filename
for os in ${oversub[@]}; do
    echo -n "$os.suv.tx" >> $filename
    echo -n "," >> $filename
    echo -n "$os.suv.rx" >> $filename
    echo -n "," >> $filename
    echo -n "$os.ac.tx" >> $filename
    echo -n "," >> $filename
    echo -n "$os.ac.rx" >> $filename
    echo -n "," >> $filename
    echo -n "$os.sc.tx" >> $filename
    echo -n "," >> $filename
    echo -n "$os.sc.rx" >> $filename
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
        txsuv=$(grep "total TX PCIe" suv.${os}.1.txt | awk {'print $5}')
        echo "$txsuv"
        rxsuv=$(grep "total RX PCIe" suv.${os}.0.txt | awk {'print $5}')
        echo "$rxsuv"
        txac=$(grep "total TX PCIe" ac.${os}.1.txt | awk {'print $5}')
        echo "$txac"
        rxac=$(grep "total RX PCIe" ac.${os}.0.txt | awk {'print $5}')
        echo "$rxac"
        if [ $benchmark == "xsbench" ] || [ $benchmark == "bfs" ] || [ $benchmark == "bptree" ] ; then
            txsc=$(grep "total TX PCIe" uvm.${os}.1.txt | awk {'print $5}')
            echo "$txsc"
            rxsc=$(grep "total RX PCIe" uvm.${os}.0.txt | awk {'print $5}')
            echo "$rxsc"
        else
            txsc=$(grep "total TX PCIe" sc.${os}.1.txt | awk {'print $5}')
            echo "$txsc"
            rxsc=$(grep "total RX PCIe" sc.${os}.0.txt | awk {'print $5}')
            echo "$rxsc"
        fi
        cd ${pwd0}
        echo -n $txsuv >> $filename
        echo -n "," >> $filename
        echo -n "$rxsuv" >> $filename
        echo -n "," >> ${filename}
        echo -n $txac >> $filename
        echo -n "," >> $filename
        echo -n "$rxac" >> $filename
        echo -n "," >> ${filename}
        echo -n $txsc >> $filename
        echo -n "," >> $filename
        echo -n "$rxsc" >> $filename
        echo -n "," >> ${filename}
        echo ""
    done
done


