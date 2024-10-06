/*********************************************************************************/
//
// Polybench kernels implementation on CUDA GPU
//
// Computer & Information Science, University of Delaware
// Author(s):   Sudhee Ayalasomayajula (sudhee1@gmail.com)
//              John Cavazos (cavazos@cis.udel.edu)
//		Scott Grauer Gray(sgrauerg@gmail.com)
//              Robert Searles (rsearles35@aol.com)   
//              Lifan Xu (xulifan@udel.edu)
//
// Contact(s): Lifan Xu (xulifan@udel.edu)
// Reference(s):
//
/*********************************************************************************/

#define MiB 18399
#define RESERVATION ((MiB*1024UL*1024))
#define SABY 0

#include "penguin.h"

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include "polybenchUtilFuncts.h"

/* #include "../../common/polybenchUtilFuncts.h" */
/* #include "../../common/polybench.h" */
/* #include "../../common/polybench.c" */

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size. */
#define NR 1024ULL
#define NQ 1024ULL
#define NP 1024ULL

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

#define GPU_DEVICE 0

int do_saby = 0;

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void doitgenCPU(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4)
{
	for (int r = 0; r < NR; r++)
	{
		for (int q = 0; q < NQ; q++)  
		{
			for (int p = 0; p < NP; p++)  
			{
				sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
				for (int s = 0; s < NP; s++)
				{
					sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
				}
      		}
      		
			for (int p = 0; p < NP; p++)
       		{
				A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
			}
		}
	}
}


void init_array(DATA_TYPE *A, DATA_TYPE *C4, DATA_TYPE *sum)
{
  	for (int i = 0; i < NR; i++)
  	{
    		for (int j = 0; j < NQ; j++)
    		{
      			for (int k = 0; k < NP; k++)
      			{
	 			A[i * (NQ * NP) + j * NP + k] = ((DATA_TYPE) i*j + k) / NP;
        sum[i * (NQ * NP) + j * NP + k] = (DATA_TYPE) 0.0;
      			}
    		}
  	}

  	for (int i = 0; i < NP; i++)
  	{
    		for (int j = 0; j < NP; j++)
    		{
      			C4[i * NP + j] = ((DATA_TYPE) i*j) / NP;
    		}
  	}
}


void compareResults(DATA_TYPE* sum, DATA_TYPE* sum_outputFromGpu)
{
	int fail = 0;
	
	for (int r = 0; r < NR; r++)
	{
    		for (int q = 0; q < NQ; q++)  
		{
      			for (int p = 0; p < NP; p++)  
			{
				if (percentDiff(sum[r * (NQ * NP) + q * NP + p], sum_outputFromGpu[r * (NQ * NP) + q * NP + p]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
		}
	}
	
	// Print results
	printf("Number of misses: %d\n", fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void doitgen_kernel1(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < NP) && (q < NQ))
	{

    // changes by Pratheek, reverted now
		sum[r * (NQ * NP) + q * NP + p] = (DATA_TYPE)0.0;
		/* DATA_TYPE temp = (DATA_TYPE)0.0; */
	
		for (int s = 0; s < NP; s++)
		{
		sum[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p] + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p];
		/* temp = temp + A[r * (NQ * NP) + q * NP + s] * C4[s * NP + p]; */
		}
		/* sum[r * (NQ * NP) + q * NP + p] = temp; */
	}
}

__global__ void doitgen_kernel2(DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if ((p < NP) && (q < NQ))
	{
		A[r * (NQ * NP) + q * NP + p] = sum[r * (NQ * NP) + q * NP + p];
	}
}

void doitgenCuda(DATA_TYPE* A, DATA_TYPE* C4, DATA_TYPE* sum)
{
	double t_start, t_end;

	DATA_TYPE* AGpu = A;
	DATA_TYPE* C4Gpu = C4;
	DATA_TYPE* sumGpu = sum;

	/* cudaMallocManaged(&AGpu, NR * NQ * NP * sizeof(DATA_TYPE)); */
	/* cudaMallocManaged(&C4Gpu, NP * NP * sizeof(DATA_TYPE)); */
	/* cudaMallocManaged(&sumGpu, NR * NQ * NP * sizeof(DATA_TYPE)); */

	/* cudaMemcpy(AGpu, A, NR * NQ * NP * sizeof(DATA_TYPE), cudaMemcpyHostToDevice); */
	/* cudaMemcpy(C4Gpu, C4, NP * NP * sizeof(DATA_TYPE), cudaMemcpyHostToDevice); */
	/* cudaMemcpy(sumGpu, sum, NR * NQ * NP * sizeof(DATA_TYPE), cudaMemcpyHostToDevice); */

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)NP) / ((float)block.x) ), (unsigned int)ceil( ((float)NR) / ((float)block.y) ));
	t_start = rtclock();
	
#if SABY
  cudaMemAdvise(A, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0);
  cudaMemAdvise(C4, sizeof(DATA_TYPE) * NQ * NP, cudaMemAdviseSetAccessedBy, 0);
  cudaMemAdvise(sum, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0);
#endif

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  penguinSetPrioritizedLocation(C4, sizeof(DATA_TYPE) * NP * NP, 0);
  /* penguinSuperPrefetch(AGpu, 1, 0, 0); */
    /* cudaMemPrefetchAsync(AGpu + (0/512)*NR/4*NP*NQ,sizeof(DATA_TYPE)*NR/4*NP*NQ, GPU_DEVICE, 0 ); */
    /* cudaMemPrefetchAsync(AGpu, sizeof(DATA_TYPE)*NR*NP*NQ, GPU_DEVICE, 0 ); */

  /* if(do_saby) { */
  /* cudaMemAdvise(A, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(C4, sizeof(DATA_TYPE) * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(sum, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */
  /*   unsigned MB = 23860-MiB-10; */
  /*   unsigned long long available = (unsigned long long)(MB) * 1024ULL * 1024ULL; */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NQ * NP)) { */
  /*     cudaMemPrefetchAsync(C4 ,sizeof(DATA_TYPE)*(NQ)*NP, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NQ * NP)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(C4 ,available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NR * NQ * NP)) { */
  /*     cudaMemPrefetchAsync(A, sizeof(DATA_TYPE) * NR * NQ * NP, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NR * NQ * NP)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(A, available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /*   if(available >= (sizeof(DATA_TYPE) * NR * NQ * NP)) { */
  /*     cudaMemPrefetchAsync(sum ,sizeof(DATA_TYPE)*NR * NQ * NP, GPU_DEVICE,  0); */
  /*     available -= ((sizeof(DATA_TYPE) * NR * NQ * NP)); */
  /*   } else { */
  /*     cudaMemPrefetchAsync(sum ,available, GPU_DEVICE,  0); */
  /*     available = 0; */
  /*   } */
  /*   printf("av = %llu\n", available); */
  /* } */


  for (int r = 0; r < NR; r++)
  {

    doitgen_kernel1 <<<grid, block>>> (sumGpu, AGpu, C4Gpu, r);
    cudaThreadSynchronize();
    doitgen_kernel2 <<<grid, block>>> (sumGpu, AGpu, C4Gpu, r);
    cudaThreadSynchronize();
  }

	t_end = rtclock();
	fprintf(stdout, "GPU.Parser.Time: %0.6lf\n", t_end - t_start);
	
	/* cudaMemcpy(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost); */

	cudaFree(AGpu);
	cudaFree(C4Gpu);
	cudaFree(sumGpu);
}
	

int main(int argc, char *argv[])
{
  int* reservation;
  cudaMalloc((void**) &reservation, RESERVATION);
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* C4;
	DATA_TYPE* sum, *sum_outputFromGpu;


	/* A = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE)); */
	/* C4 = (DATA_TYPE*)malloc(NP * NP * sizeof(DATA_TYPE)); */
	/* sum = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE)); */
	/* sum_outputFromGpu = (DATA_TYPE*)malloc(NR * NQ * NP * sizeof(DATA_TYPE)); */

	cudaMallocManaged(&A, NR * NQ * NP * sizeof(DATA_TYPE));
	cudaMallocManaged(&C4, NP * NP * sizeof(DATA_TYPE));
	cudaMallocManaged(&sum, NR * NQ * NP * sizeof(DATA_TYPE));

  /* penguinSetPrioritizedLocation(B_gpu, sizeof(DATA_TYPE) * NK * NJ, 0); */

	init_array(A, C4, sum);

  /* cudaMemAdvise(A, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(C4, sizeof(DATA_TYPE) * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */
  /* cudaMemAdvise(sum, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemAdviseSetAccessedBy, 0); */

  penguinStartStatCollection();
  nvml_start();
	doitgenCuda(A, C4, sum);
  nvml_stop();
  penguinStopStatCollection();

	/* t_start = rtclock(); */
	/* doitgenCPU(sum, A, C4); */
	/* t_end = rtclock(); */

	/* fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start); */
	
	/* compareResults(sum, sum_outputFromGpu); */

	/* free(A); */
	/* free(C4); */
	/* free(sum); */
	/* free(sum_outputFromGpu); */
	
    return 0;
}

