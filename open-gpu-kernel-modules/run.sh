echo "make"
make modules -j4
sleep 1
echo "rm drivers"
sudo rmmod nvidia_uvm
sleep 1
sudo rmmod nvidia
sleep 1
echo "ins drivers"
sudo insmod kernel-open/nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1
sleep 1
sudo insmod kernel-open/nvidia-uvm.ko
sleep 1
watch -n 0.1 nvidia-smi
