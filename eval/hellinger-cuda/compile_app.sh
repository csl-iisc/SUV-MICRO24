#!/bin/bash

penguinpath=$1
compilerpath=$2
binary=$3

rm *.ll *.o *.ptx *.fatbin ${binary}

clang++  -O1 --cuda-gpu-arch=sm_86 -I${penguinpath} -I/usr/local/cuda-11.8/include -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -pthread  -S -emit-llvm main.cu

opt --loop-simplify -o loopsim.ll --debug-pass-manager -S main-cuda-nvptx64-nvidia-cuda-sm_86.ll

opt -enable-new-pm=0 -load ${compilerpath}/build/lib/CudaAnalysis.so  --disable-output --debug-pass-manager loopsim.ll

## analysis uses O1 for now
## proper comilation uses O3
clang++  -O3 --cuda-gpu-arch=sm_86 -I${penguinpath} -I/usr/local/cuda-11.8/include -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -pthread  -S -emit-llvm main.cu

opt --loop-simplify -o loopsim.ll --debug-pass-manager -S main-cuda-nvptx64-nvidia-cuda-sm_86.ll

llc -mcpu=sm_86 loopsim.ll -o device.ptx

ptxas --gpu-name=sm_86 device.ptx -o device.ptx.o

fatbinary -64 --create device.fatbin --image=profile=sm_86,file=device.ptx.o --image=profile=compute_86,file=device.ptx

clang -Xclang -fcuda-include-gpubinary -Xclang './device.fatbin'  --cuda-host-only -fproc-stat-report -O3  --cuda-gpu-arch=sm_86 -I${penguinpath} -I/usr/local/cuda-11.8/include  -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -lpthread -S -emit-llvm main.cu

opt -enable-new-pm=0 -load ${compilerpath}/build/lib/DynamicHostTransform.so -S -o modif.ll   --debug-pass-manager main.ll

llc --relocation-model=pic -filetype=obj modif.ll

clang++ -L/usr/local/cuda-11.8/lib64 -lcudart -ldl -lrt -lpthread -lnvidia-ml -L${penguinpath} modif.o  -o ${binary}
