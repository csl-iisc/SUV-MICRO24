#!/bin/bash

penguinpath=$1
compilerpath=$2
binary=$3

rm *.ll *.o *.ptx *.fatbin ${binary}

clang++  -O3 --cuda-gpu-arch=sm_86 -I${penguinpath} -I/usr/local/cuda-11.8/include -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -pthread  -S -emit-llvm Simulation.cu

opt --loop-simplify -o loopsim.ll --debug-pass-manager -S Simulation-cuda-nvptx64-nvidia-cuda-sm_86.ll

opt -enable-new-pm=0 -load ${compilerpath}/build/lib/CudaAnalysis.so  --disable-output --debug-pass-manager loopsim.ll

llc -mcpu=sm_86 loopsim.ll -o device.ptx
ptxas --gpu-name=sm_86 device.ptx -o device.ptx.o
fatbinary -64 --create device.fatbin --image=profile=sm_86,file=device.ptx.o --image=profile=compute_86,file=device.ptx

# compile all .cu files to .ll files
clang -Xclang -fcuda-include-gpubinary -Xclang './device.fatbin'  --cuda-host-only -fproc-stat-report -O3  --cuda-gpu-arch=sm_86 -I${penguinpath} -I/usr/local/cuda-11.8/include  -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -lpthread -S -emit-llvm *.cu

opt -enable-new-pm=0 -load ${compilerpath}/build/lib/DynamicHostTransform.so -S -o looprotated.ll  -loop-rotate --debug-pass-manager Simulation.ll
opt -enable-new-pm=0 -load ${compilerpath}/build/lib/DynamicHostTransform.so -S -o modified.ll   --debug-pass-manager looprotated.ll
opt -S -O3 -o modif.ll modified.ll

llc --relocation-model=pic -filetype=obj modif.ll
llc --relocation-model=pic -filetype=obj main.ll
llc --relocation-model=pic -filetype=obj XSutils.ll
llc --relocation-model=pic -filetype=obj Materials.ll
llc --relocation-model=pic -filetype=obj GridInit.ll
llc --relocation-model=pic -filetype=obj io.ll
clang++ -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -lpthread -lnvidia-ml modif.o main.o io.o GridInit.o Materials.o XSutils.o -o ${binary}
