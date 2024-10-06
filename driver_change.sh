#!/bin/bash

echo "Access counter Migration is ${1}"
echo "Access counter Granularity is ${2}"
echo "Access counter Threshold is ${3}"

success=1
echo "rm drivers"
while [ $success -eq 1 ]
do
  echo "rm nvidia_uvm"
  sudo rmmod nvidia_uvm
  success=$?
  sleep 1
done
success=1
while [ $success -eq 1 ]
do
  echo "rm nvidia"
  sudo rmmod nvidia
  success=$?
  sleep 1
done
echo "ins drivers"
sudo insmod /home/pratheek/projects/micro24-ae/open-gpu-kernel-modules/kernel-open/nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1
sleep 1
sudo insmod /home/pratheek/projects/micro24-ae/open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko uvm_perf_access_counter_mimc_migration_enable=${1} uvm_perf_access_counter_granularity=${2} uvm_perf_access_counter_threshold=${3}
sleep 1
sudo lsmod | grep -i nvidia
# watch -n 0.1 nvidia-smi
