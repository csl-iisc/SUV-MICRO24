/home/pratheek/storage/pratheek/llvm-project-llvmorg-15.0.5/build/bin/clang++   -I. -O3 -DNDEBUG --cuda-gpu-arch=sm_52 --cuda-path=/usr/local/cuda-11.8 -g -Xcompiler=-Wall -lineinfo -std=c++17 -MD -MT test_map_search.cu.o -MF test_map_search.cu.o.d -x cuda -c test_map_search.cu -o test_map_search.cu.o

/home/pratheek/storage/pratheek/llvm-project-llvmorg-15.0.5/build/bin/clang++ test_map_search.cu.o -o test_search  -lstdc++fs -lcudadevrt -lcudart_static -lrt -lpthread -ldl  -L"/usr/local/cuda-11.8/lib64"
