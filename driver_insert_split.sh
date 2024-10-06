#!/bin/bash

echo "ins drivers"
sudo insmod $SUVHOME/open-gpu-kernel-modules/kernel-open/nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1
sleep 1
sudo insmod $SUVHOME/open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko 
sleep 1
sudo lsmod | grep -i nvidia
