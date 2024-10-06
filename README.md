# Description of the artifact folder
------------------------------------

1. The folder 'llvm' contains LLVM 15 and our passes.
2. The header file 'penguin.h' contains our runtime components.
3. The folder 'eval' contains workloads.
4. The fodler open-gpu-kernel-modules contains the UVM driver.
5. The script compile.sh compiles everything
6. The script run.sh runs everything.

# Prerequisites for LLVM
------------------------

Please install prerequisites as mentioned in LLVM website.

# Compile and build LLVM and SUV passes
---------------------------------------

```
cd llvm
mkdir build
cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS='clang' -DLLVM_ENABLE_BACKENDS='x86;nvptx' -DCMAKE_BUILD_TYPE='Debug'
cd bulid
ninja
```

Compiling LLVM can take between 20 minutes to 2 hours, depending on the amount of compute and memory available.
In machines with limted memory, the linking process may fail initially (due to out of memory).
Please run ninja again if this happens.


# Compile and insert the UVM driver
------------------------

From the top-level folder,

```
cd open-gpu-kernel-modules
make modules -j
sudo rmmod nvidia_uvm
sudo rmmod nvidia
sudo insmod /home/pratheek/projects/accesscounter/open-gpu-kernel-modules/kernel-open/nvidia.ko NVreg_OpenRmEnableUnsupportedGpus=1
sudo insmod /home/pratheek/projects/accesscounter/open-gpu-kernel-modules/kernel-open/nvidia-uvm.ko
```

# Path setting
--------------

Edit the startup.sh file to point the bin directory in LLVM.
Source the startup.sh file add the LLVM binaries into the path.

# Compile the binaries

Run the provided compile.sh script to compile all the workloads for all the configurations.
The script internally calls other scripts to set various memory reservation amounts in the main.cu file of each workload.

# Run the workloads

Use the provided run.sh to run all the compiled binaries and generate .txt for the primary graph.
Use the provided parse.sh script to parse the output of the workloads into a csv file.

# Extending SUV

SUV's passes are located in llvm/llvm/lib/Transforms/CudaAnalysis and llvm/llvm/lib/Transforms/DynamicHostTransform.
The runtime components (SUV's default policies) are located in the header file penguin-suv.h.

