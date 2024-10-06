/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "penguin.h"
#include "polybenchUtilFuncts.h"

#define GPU_DEVICE 0

#define MiB 19252
#define RESERVATION ((MiB*1024UL*1024))

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 512 * 48
#define NJ 512 * 48
#define NK 512 * 48

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

int do_soln = 0;
int do_saby = 0;

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *A_gpu, DATA_TYPE *B_gpu, DATA_TYPE *C_gpu)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
			A_gpu[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
			  B_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
			  C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
			  C_gpu[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}


/* void gemmCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu) */
void gemmCuda(DATA_TYPE* A_gpu, DATA_TYPE* B_gpu, DATA_TYPE* C_gpu, DATA_TYPE* B)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NJ)/ ((float)block.x) )),(size_t)(ceil( ((float)NI)/ ((float)block.y) )));

	t_start = rtclock();

	/* cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NK * NI, cudaMemcpyHostToDevice); */
	/* cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice); */

  /* cudaMemPrefetchAsync(A_gpu,NK*NK*sizeof(DATA_TYPE), GPU_DEVICE, 0 ); */
  /* cudaMemPrefetchAsync(B_gpu,NK*NK*sizeof(DATA_TYPE), GPU_DEVICE, 0 ); */
  /* cudaMemPrefetchAsync(C_gpu,NK*NK*sizeof(DATA_TYPE), GPU_DEVICE, 0 ); */

  /* penguinSetPrioritizedLocation(A_gpu, 0, 0); */
  /* penguinSetNoMigrateRegion(A_gpu, 0, 0, false); */

  /* cudaMemAdvise(A_gpu, sizeof(DATA_TYPE) * NI * NK, cudaMemAdviseSetPreferredLocation, 0); */
  /* cudaMemAdvise(B_gpu, sizeof(DATA_TYPE) * NK * NJ, cudaMemAdviseSetPreferredLocation, 0); */
  /* cudaMemAdvise(C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemAdviseSetPreferredLocation, CU_DEVICE_CPU); */

  /* cudaMemAdvise(A_gpu, sizeof(DATA_TYPE) * NI * NK, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(B_gpu, sizeof(DATA_TYPE) * NK * NJ, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemAdviseSetAccessedBy, 0); */
  /* if(do_saby) { */
  /* cudaMemAdvise(A_gpu, sizeof(DATA_TYPE) * NI * NK, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(B_gpu, sizeof(DATA_TYPE) * NK * NJ, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemAdviseSetAccessedBy, 0); */
  /*   unsigned MB = 23860-MiB-10; */
  /*   unsigned long long available = (unsigned long long)(MB) * 1024ULL * 1024ULL; */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NI * NI)) { */
  /*     cudaMemPrefetchAsync(C_gpu ,sizeof(DATA_TYPE)*(NI)*NI, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NI * NI)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(C_gpu ,available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NI * NI)) { */
  /*     cudaMemPrefetchAsync(A_gpu ,sizeof(DATA_TYPE)*(NI)*NI, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NI * NI)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(A_gpu ,available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NI * NI)) { */
  /*     cudaMemPrefetchAsync(B_gpu ,sizeof(DATA_TYPE)*(NI)*NI, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NI * NI)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(B_gpu ,available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /* } */

  /* if(do_soln) { */
  /*   /1* cudaMemAdvise(C_gpu, sizeof(int) * N * N, cudaMemAdviseSetAccessedBy, 0); *1/ */
  /*   cudaMemPrefetchAsync(B_gpu ,(sizeof(int)*NK*NJ * 16)/16, 0,  0); */
  /*   penguinSetPrioritizedLocation(B_gpu, (sizeof(int) * NK * NJ * 16)/16, 0); */
  /*   /1* cudaMemAdvise(B_gpu + ((NK * NJ * 15)/16), sizeof(int) * NK * NJ / 16, cudaMemAdviseSetAccessedBy, 0); *1/ */
  /*   /1* penguinSetNoMigrateRegion(d_c, sizeof(int) * N *N, 0, true); *1/ */
  /* } */

  penguinStartStatCollection();
  nvml_start();
	gemm_kernel<<< grid, block >>>(A_gpu, B_gpu, C_gpu);
	cudaDeviceSynchronize();
  nvml_stop();
  penguinStopStatCollection();

	t_end = rtclock();
	fprintf(stdout, "GPU.Parser.Time: %0.6lf\n", t_end - t_start);   

}
	

int main(int argc, char *argv[])
{

  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
  /* cudaMalloc((void**) &reservation, (22*1024UL*1024*1024) + (512 *1024UL*1024)); */
  /* cudaMalloc((void**) &reservation, (22*1024UL*1024*1024) ); */

	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C; 
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	/* cudaMalloc(&A_gpu, sizeof(DATA_TYPE) * NI * NK); */
	/* cudaMalloc(&B_gpu, sizeof(DATA_TYPE) * NK * NJ); */
	cudaMallocManaged(&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMallocManaged(&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

	init(A, B, C, A_gpu, B_gpu, C_gpu);

	GPU_argv_init();
	
  /* nvml_start(); */
	/* gemmCuda(A_gpu, B_gpu, C_gpu); */
	gemmCuda(A_gpu, B_gpu, C_gpu, B);
  /* nvml_stop(); */

	/* t_start = rtclock(); */	
	/* gemm(A, B, C); */
	/* t_end = rtclock(); */
	/* fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); */
	
	compareResults(C, C_gpu);

	free(A);
	free(B);  
	free(C);  
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
    return 0;
}

