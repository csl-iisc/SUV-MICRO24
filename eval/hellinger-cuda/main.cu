#include <iostream>
#include <new>
#include <cmath>
#include <chrono>
#include <cuda.h>

#define BLOCK_SIZE 16

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif

#include "penguin.h"

#define MiB 19252
#define RESERVATION ((MiB*1024UL*1024))

int do_soln_ac = 0;
int do_soln_ze = 0;
/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 512 * 8 * 24;  // Must be a multiple of 8.
constexpr int M = m_size / 4; // 8
constexpr int N = m_size / 4; // 4
constexpr int P = m_size / 4; // 2

#ifdef VERIFY
#include "verify.h"
#endif

__global__ 
void hellinger(
  const FP *__restrict__ a, 
  const FP *__restrict__ b, 
        FP *__restrict__ c, 
  const int m, const int n, const int k)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    /* if( col < k && row < m) */
    {
        FP sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += SQRT(a[row * n + i] * b[i * k + col]);
        }
        const FP value = (FP)1.0 - sum;
        const FP gate = (!signbit(value));
        c[row * k + col] = SQRT(gate * value);
    }
}

int main(int argc, char** argv)
{
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
  /* if (argc != 2) { */
  /*   printf("Usage: %s <repeat>\n", argv[0]); */
  /*   return 1; */
  /* } */
  const int repeat = 1;

  int i, j;

  // 2D arrays on host side.
  FP(*a_host)[N] = new FP[M][N];
  FP(*b_host)[P] = new FP[N][P];
  // host-side cpu result
  FP(*c_host)[P] = new FP[M][P];
  // host-side gpu result
  FP(*c_back)[P] = new FP[M][P];

  FP *a_device, *b_device, *c_device;

  cudaMallocManaged((void **) &a_device, sizeof(FP)*M*N);
  cudaMallocManaged((void **) &b_device, sizeof(FP)*N*P);
  cudaMallocManaged((void **) &c_device, sizeof(FP)*M*P);

  std::cout << a_device << std::endl;
  std::cout << b_device << std::endl;
  std::cout << c_device << std::endl;

  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      a_device[i*N+j] = (FP)1.0 / N;

  srand(123);
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++)
      b_device[i*P+j] = rand() % 256;

  for (j = 0; j < P; j++) { 
    FP sum = 0;
    for (i = 0; i < N; i++)
      sum += b_device[i*P+j];
    for (i = 0; i < N; i++)
      b_device[i*P+j] /= sum;
  }

  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++)
      c_device[i*P+j] = 0.0;


  /* cudaMemcpy(a_device, a_host, sizeof(FP)*M*N, cudaMemcpyHostToDevice); */
  /* cudaMemcpy(b_device, b_host, sizeof(FP)*N*P, cudaMemcpyHostToDevice); */

  unsigned int grid_cols = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  nvml_start();
  penguinStartStatCollection();
  /* if(do_soln_ac) { */
  /*   /1* cudaMemAdvise(a_device, sizeof(int) * N * N, cudaMemAdviseSetAccessedBy, 0); *1/ */
  /*   cudaMemAdvise(b_device, sizeof(int) * N * N, cudaMemAdviseSetAccessedBy, 0); */
  /*   cudaMemAdvise(c_device, sizeof(int) * N * N, cudaMemAdviseSetAccessedBy, 0); */
  /*   /1* cudaMemPrefetchAsync(a_device ,sizeof(int)*(N)*N, 0,  0); *1/ */
  /*   cudaMemPrefetchAsync(b_device ,sizeof(int)*(N)*N, 0,  0); */
  /*   penguinSetPrioritizedLocation(b_device, sizeof(int) * N * N, 0); */
  /*   penguinSetQuickMigrate(a_device, sizeof(int) * N * N, true); */
  /*   /1* penguinSetNoMigrateRegion(c_device, sizeof(int) * N * N, 0, true); *1/ */
  /* } */
  /* if(do_soln_ze) { */
  /*   cudaMemPrefetchAsync(a_device ,sizeof(int)*(N)*N, 0,  0); */
  /*   cudaMemPrefetchAsync(b_device ,sizeof(int)*(N)*N, 0,  0); */
  /*   cudaMemPrefetchAsync(c_device ,sizeof(int)*(N)*N, 0,  0); */
  /* } */

  std:: cout<< "hello\n";
    /* penguinStartStatCollection(); */
  /* for (int i = 0; i < repeat; i++) */
    hellinger<<<dimGrid, dimBlock>>>(a_device, b_device, c_device, M, N, P);

  cudaDeviceSynchronize();
    nvml_stop();
  penguinStopStatCollection();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "GPU.Parser.Time: " << (time * 1e-9f) / repeat << "\n";

  cudaMemcpy(c_back, c_device, sizeof(int)*M*P, cudaMemcpyDeviceToHost);

  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
            << ") * b(" << N << "," << P << ")\n";

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(c_device);
  return 0;
}
