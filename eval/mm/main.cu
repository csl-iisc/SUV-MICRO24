#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "penguin.h"

#define MiB 20020
#define RESERVATION ((MiB*1024UL*1024))

int do_saby = 0;

int do_soln_ac = 0;

#define SABY 0
#define VANILLA 0
#define SOLUTION 0

/* Problem size */
#define N 512ULL * 48ULL

/* #define BLOCK_SIZE 16 */
#define GRID_SIZE (N/BLOCK_SIZE)

#define BLOCK_SIZE 32
__global__ void matrixMultiplyKernel(float *C, float *A, float *B,
                                     /* unsigned int matrixDim) { */
                                     unsigned int wA, unsigned int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
  for (int i = 0; i < m; ++i) 
  {
    for (int j = 0; j < k; ++j) 
    {
      int tmp = 0.0;
      for (int h = 0; h < n; ++h) 
      {
        tmp += h_a[i * n + h] * h_b[h * k + j];
      }
      h_result[i * k + j] = tmp;
    }
  }
}

int main(int argc, char const *argv[])
{
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);

  int m, n, k;
  /* Fixed seed for illustration */
  srand(3333);
  /* printf("please type in m n and k\n"); */
  /* scanf("%d %d %d", &m, &n, &k); */
  m = n = k = N;
  m = 36 * 512ULL;
  n = 48 * 512ULL;
  k = 48 * 512ULL;

  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start to count execution time of GPU version
  // Allocate memory space on the device 
  float *d_a, *d_b, *d_c;
  cudaMallocManaged((void **) &d_a, sizeof(int)*m*n);
  cudaMallocManaged((void **) &d_b, sizeof(int)*n*k);
  cudaMallocManaged((void **) &d_c, sizeof(int)*m*k);

  float *a,*b;

  // random initialize matrix A
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      /* a[i * N + j] = rand() % 1024; */
      d_a[i * N + j] = rand() % 1024;
    }
  }

  // random initialize matrix B
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      /* b[i * k + j] = rand() % 1024; */
      d_b[i * N + j] = rand() % 1024;
    }
  }

  // random initialize matrix C
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      /* c[i * k + j] = rand() % 1024; */
      /* d_c[i * N + j] = 0; */
    }
  }

  unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    nvml_start();
  cudaEventRecord(start, 0);

  penguinPinHost(d_c, 0);
  // Launch kernel 
    penguinStartStatCollection();
  /* cudaProfilerStart(); */
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, n, k);    
  /* cudaProfilerStop(); */
  cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute time elapse on GPU computing
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
  printf("GPU.Parser.Time: %f\n", gpu_elapsed_time_ms);

  penguinStopStatCollection();
    nvml_stop();

  // free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  /* cudaFreeHost(h_a); */
  /* cudaFreeHost(h_b); */
  /* cudaFreeHost(h_c); */
  /* cudaFreeHost(h_cc); */
  return 0;
}
