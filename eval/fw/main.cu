#include <sys/stat.h> // stat
#include <unistd.h> // getopt
#include <chrono> // high_resolution_clock
#include <iostream> // cout
#include <vector> // cout
#include <cstdio> // printf
#include <fstream> // ifstream, ofstream
#include <sstream> // stringstream
#include <ratio>  // milli
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "floyd_warshall.hpp"
#include "penguin.h"

#define BLOCK_DIM 16
#define CUDA

#define MiB 21130
#define RESERVATION ((MiB*1024UL*1024))

int do_saby = 0;

__forceinline__
__host__ void check_cuda_error() {
  cudaError_t errCode = cudaPeekAtLastError();
  if (errCode != cudaSuccess) {
    std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," <<
                cudaGetErrorString(errCode) << "\n";
  }
}

__forceinline__
__device__ void calc(int* graph, int n, int k, int i, int j) {
  if ((i >= n) || (j >= n) || (k >= n)) return;
  const unsigned int kj = k*n + j;
  const unsigned int ij = i*n + j;
  const unsigned int ik = i*n + k;
  int t1 = graph[ik] + graph[kj];
  int t2 = graph[ij];
  graph[ij] = (t1 < t2) ? t1 : t2;
}


__global__ void floyd_warshall_kernel(int n, int k, int* graph) {
  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  calc(graph, n, k, i, j);
}

/*****************************************************************************
                         Blocked Floyd-Warshall Kernel
  ***************************************************************************/

__forceinline__
__device__ void block_calc(int* C, int* A, int* B, int bj, int bi) {
  for (int k = 0; k < BLOCK_DIM; k++) {
    int sum = A[bi*BLOCK_DIM + k] + B[k*BLOCK_DIM + bj];
    if (C[bi*BLOCK_DIM + bj] > sum) {
      C[bi*BLOCK_DIM + bj] = sum;
    }
    __syncthreads();
  }
}

__global__ void floyd_warshall_block_kernel_phase1(unsigned n, int k, int* graph) {
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  // Transfer to temp shared arrays
  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();
  
  block_calc(C, C, C, bi, bj);

  __syncthreads();

  // Transfer back to graph
  graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

}


__global__ void floyd_warshall_block_kernel_phase2(unsigned n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int i = blockIdx.x;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k) return;

  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, C, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];

  // Phase 2 1/2

  C[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, C, bi, bj);

  __syncthreads();

  // Block C is the only one that could be changed
  graph[k*BLOCK_DIM*n + i*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}


__global__ void floyd_warshall_block_kernel_phase3(unsigned n, int k, int* graph) {
  // BlockDim is one dimensional (Straight along diagonal)
  // Blocks themselves are two dimensional
  const unsigned int j = blockIdx.x;
  const unsigned int i = blockIdx.y;
  const unsigned int bi = threadIdx.y;
  const unsigned int bj = threadIdx.x;

  if (i == k && j == k) return;
  __shared__ int A[BLOCK_DIM * BLOCK_DIM];
  __shared__ int B[BLOCK_DIM * BLOCK_DIM];
  __shared__ int C[BLOCK_DIM * BLOCK_DIM];

  __syncthreads();

  C[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];
  A[bi*BLOCK_DIM + bj] = graph[i*BLOCK_DIM*n + k*BLOCK_DIM + bi*n + bj];
  B[bi*BLOCK_DIM + bj] = graph[k*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj];

  __syncthreads();

  block_calc(C, A, B, bi, bj);

  __syncthreads();

  graph[i*BLOCK_DIM*n + j*BLOCK_DIM + bi*n + bj] = C[bi*BLOCK_DIM + bj];
}

/************************************************************************
                    Floyd-Warshall's Algorithm CUDA
************************************************************************/


__host__ void floyd_warshall_blocked_cuda(int* input, int* output, unsigned n, int* device_graph) {

  n = 32 * 1024; // remove this
  /* int deviceCount; */
  /* cudaGetDeviceCount(&deviceCount); */

  /* for (int i = 0; i < deviceCount; i++) { */
  /*   cudaDeviceProp deviceProps; */
  /*   cudaGetDeviceProperties(&deviceProps, i); */

    /* std::cout << "Device " << i << ": " << deviceProps.name << "\n" */
	      /* << "\tSMs: " << deviceProps.multiProcessorCount << "\n" */
	      /* << "\tGlobal mem: " << static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024 * 1024) << "GB \n" */
	      /* << "\tCUDA Cap: " << deviceProps.major << "." << deviceProps.minor << "\n"; */
  /* } */

  /* int* device_graph; */
  /* const size_t size = sizeof(int) * n * n; */
  /* const size_t size = sizeof(int) * 1024 * 1024ULL * 1024ULL; */
  /* cudaMallocManaged(&device_graph, size); */
  /* /1* cudaMemcpy(device_graph, input, size, cudaMemcpyHostToDevice); *1/ */
  /* memcpy(device_graph, input, size); */

  const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
  dim3 block_dim(BLOCK_DIM, BLOCK_DIM, 1);
  dim3 phase4_grid(blocks, blocks, 1);

  /* if(do_saby) { */
  /*   cudaMemAdvise(device_graph, sizeof(int) * n * n, cudaMemAdviseSetAccessedBy, 0); */
  /*     /1* cudaMemPrefetchAsync(device_graph , sizeof(int) * n * n, 0,  0); *1/ */
  /*     cudaMemPrefetchAsync(device_graph , (23860-MiB)*1024ULL*1024ULL, 0, 0); */
  /* } */

    /* cudaMemAdvise(device_graph, sizeof(int) * n * n, cudaMemAdviseSetAccessedBy, 0); */

  std::cout << "Launching Kernels Blocks: " << blocks << " Size " << n << "\n";
  for (int k = 0; k < blocks; k++) {
    /* if(k % (blocks/2) == 0){ */
      /* if(k == 0) { */
      /*   /1* std::cout << "k = 0" << "\n"; *1/ */
      /*   /1* cudaMemPrefetchAsync(device_graph, sizeof(int) * n * 1 * n/2, 0,  0); *1/ */
      /*   unsigned long long size = (n * n * 50ULL) / 100; */
      /*   cudaMemPrefetchAsync(device_graph, sizeof(int) * size, 0,  0); */
      /*   /1* for (int rowi = 0; rowi < 32768; rowi++) { *1/ */
      /*   /1*   cudaMemPrefetchAsync(device_graph + priter*8192 + rowi*32768, 32768, 0,  0); *1/ */
      /*   } */
      /* } */
      /* if(k == 819) { */
      /* if(k == 1365) { */
      /*   std::cout << "k = 1365" << "\n"; */
      /*   /1* cudaMemPrefetchAsync(device_graph+n*1*n/2, sizeof(int) * n * n/2, 0,  0); *1/ */
      /*   cudaMemPrefetchAsync(device_graph+n*2*n/3, sizeof(int) * n * 1* n/3, 0,  0); */
      /* check_cuda_error(); */
      /* } */
      /* if(k == (2048 * 100)/150 ) { */
      /*   std::cout << "k = " << "\n"; */
      /*   unsigned long long size = (n * n * 50ULL) / 100; */
      /*   unsigned long long start = (n * n * 50ULL) / 100; */
      /*   std::cout << start << " " << size << "\n"; */
      /*   cudaMemPrefetchAsync(device_graph+start, sizeof(int) * size, 0,  0); */
      /* } */
    /* } */
    floyd_warshall_block_kernel_phase1<<<1, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase2<<<blocks, block_dim>>>(n, k, device_graph);

    floyd_warshall_block_kernel_phase3<<<phase4_grid, block_dim>>>(n, k, device_graph);
    /* cudaThreadSynchronize(); */
  }
  
  /* cudaMemcpy(output, device_graph, size, cudaMemcpyDeviceToHost); */
  check_cuda_error();

  cudaFree(device_graph);
}

int* floyd_warshall_blocked_init(const int n, const int block_size, const double p, const unsigned long seed) {
  static std::uniform_real_distribution<double> flip(0, 1);
  // TODO: create negative edges without negative cycles
  static std::uniform_int_distribution<int> choose_weight(1, 100);

  std::mt19937_64 rand_engine(seed);

  int n_oversized;
  int block_remainder = n % block_size;
  if (block_remainder == 0) {
    n_oversized = n;
  } else {
    n_oversized = n + block_size - block_remainder;
  }

  int* out = new int[n_oversized * n_oversized];
  for (int i = 0; i < n_oversized; i++) {
    for (int j = 0; j < n_oversized; j++) {
      if (i == j) {
        out[i*n_oversized + j] = 0;
      } else if (i < n && j < n && flip(rand_engine) < p) {
        out[i*n_oversized + j] = choose_weight(rand_engine);
      } else {
        // "infinity" - the highest value we can still safely add two infinities
        out[i*n_oversized + j] = std::numeric_limits<int>::max() / 2;
      }
    }
  }

  return out;
}

int main(int argc, char* argv[]) {
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
  unsigned long seed = 0;
  unsigned long n = 32 * 1024;
  double p = 0.5;
  int block_size = 32;
  int thread_count = 1;
  int* matrix = nullptr;
  int* output = nullptr;
  matrix = floyd_warshall_blocked_init(n, block_size, p, seed);
  /* unsigned long long n_blocked = n; */
  unsigned long n_blocked = n;
  int block_remainder = n % block_size;
  if (block_remainder != 0) {
    n_blocked = n + block_size - block_remainder;
  }
  output = new int[n_blocked * n_blocked];

  nvml_start();
  penguinStartStatCollection();
  std::cout << "Using Floyd-Warshall's on " << n_blocked << "x" << n_blocked
    << " with p=" << p << " and seed=" << seed << "\n";
  auto start = std::chrono::high_resolution_clock::now();
#ifdef CUDA
  int* device_graph;
  const size_t size = sizeof(int) * 1024 * 1024ULL * 1024ULL;
  cudaMallocManaged(&device_graph, size);
  memcpy(device_graph, matrix, size);
  floyd_warshall_blocked_cuda(matrix, output, n_blocked, device_graph);
#else
  floyd_warshall_blocked(matrix, output, n_blocked, block_size);
#endif
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> start_to_end = end - start;
  std::cout << "GPU.Parser.Time: " << start_to_end.count() << "\n\n";
  nvml_stop();
  penguinStopStatCollection();

  delete[] matrix;
  delete[] output;
}
