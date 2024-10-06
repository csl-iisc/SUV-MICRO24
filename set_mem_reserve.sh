#!/bin/bash

GPU_SIZE=23860
benchmark=$1
footprint=$2
oversub=$3

# assuming that 
echo "set reserve"
echo $(pwd)
echo ${benchmark} ${footprint}
cd eval
cd $benchmark
available=$((footprint*100/(100+oversub)))
reservation=$((GPU_SIZE-available))
echo $reservation
sed -i "s/#define MiB.*/#define MiB ${reservation}/g" main.cu
cd ../..
# available=$((footprint*100/(100+oversub)))
available=$((available-10))
echo $available

sed -i "s/unsigned long long MBs.*/unsigned long long MBs = ${available}ULL;/g" penguin.h
