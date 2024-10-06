/**********************************************************************
* FILENAME :        alexnet_host.cu             
* 
* DESCRIPTION :
*       Host side implementation of AlexNet network
*
* NOTES :
*       This file includes CUDA memory allocations and CUDA
*       memory copies to host.
*       Invokes kernel from host.
*       Reads inputs and weight from files
* 
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <time.h>
// includes, project
//#include <cutil.h>
#define L1_KERNEL_SIZE 11*11*3
#define L1_OUT 96
#define L2_KERNEL_SIZE 5*5*48
#define L2_OUT 256 
#define L3_KERNEL_SIZE 3*3*256
#define L3_OUT 384 
#define L4_KERNEL_SIZE 3*3*192
#define L4_OUT 384
#define L5_KERNEL_SIZE 3*3*192
#define L5_OUT 256
#define INPUT_SIZE 227*227*3

#define L1_FMAP 55*55
#define L2_FMAP 27*27
#define L3_FMAP 13*13
#define L4_FMAP 13*13
#define L5_FMAP 13*13
#define POOL1_FMAP 27*27
#define POOL2_FMAP 13*13
#define POOL3_FMAP 6*6
//#define CPU
// includes, kernels
#include "an_kernel.cu"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C"
void NeuralNetwork();
unsigned g_verbose;
#define MiB 21527
#define NUM 2047

#include "penguin.h"

bool isClose(double a, double b) {
    if (abs((a-b)/b) < 0.01)
        return true;
    return false;
}

void debuggy(float* array, unsigned size) {
    unsigned count = 0;
    for(int i = 0; i < size; i++) {
        if(array[i] != 0){
            count++;
            /* std::cout << i  << " "; */
            /* std::cout << array[0+i] << " " << array[4*size+i]; */
            /* std::cout << std::endl; */
            if(!isClose(array[0+i], array[4*size+i])) {
                std::cout << "mismatch\n";
            std::cout << array[0+i] << " " << array[4*size+i];
            std::cout << std::endl;
            }
        }
    }
    std::cout << "non zero count = " << count << " out of " << size << std::endl;
}

void extract_weights(const char *pFileName,float *layer_weights,bool bias)
{
    FILE * pFile1 = fopen (pFileName,"rb");
    char delim[2];
    if(bias == true)
        delim[0] = ' ';
    else
        delim[0] = '\n';
    delim[1] = 0;
    char *token;
    int count = 0;
    char *line = NULL;
    size_t len = 0;
    if (!(pFile1 != NULL))
        printf("File Not Found\n");
    if (pFile1 != NULL && (bias == false))
    {
        printf(" File FOUND %s\n",pFileName);
        {

            //fread(weights,sizeof(weights),1,pFile1);
            //token = strtok(weights,delim);
            //while(token != NULL)
            while (getline(&line, &len, pFile1) != -1)
            {
                token = strtok(line,delim);
                float temp_num = atof(token);
                layer_weights[count] = temp_num;	
                //printf("%.8f\t",temp_num); 
                count++; 
                //	token = strtok(NULL,delim);
            }
        }
        printf("Final Count : %d\n",count);
        fclose(pFile1);
    }
    if (pFile1 != NULL && (bias == true))
    {
        printf(" File FOUND %s\n",pFileName);
        {

            char weights[94590] = "";
            fread(weights,sizeof(weights),1,pFile1);
            token = strtok(weights,delim);
            while(token != NULL)
            {
                float temp_num = atof(token);
                layer_weights[count] = temp_num;	
                //printf("%.8f\t",temp_num); 
                count++; 
                token = strtok(NULL,delim);
            }
        }
        printf("Final Count : %d\n",count);
        fclose(pFile1);
    }

}

int main(int argc, char** argv)
{
    int *reservation;
    int i, commandline_error;
    commandline_error = 0;
    g_verbose = 0;
    /* if (argc >= 2) { */
        /* NUM = atoi(argv[1]); */
        /* MiB = atoi(argv[2]); */
        /* MiB = 23860 - MiB; */
    /* } else commandline_error=1; */
    /* if (commandline_error || !NUM) { */
    /*     printf("Usage: ./AN <NUM> [-v]\n"); */
    /*     printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n"); */
    /*     return 1; */
    /* } */
    cudaMalloc(&reservation, MiB*1024ULL*1024ULL);
    NeuralNetwork();
}
void Fill_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU)
{
    extract_weights("data/conv1.txt",Layer1_Weights_CPU,false);
    extract_weights("data/conv2.txt",Layer2_Weights_CPU,false);
    extract_weights("data/conv3.txt",Layer3_Weights_CPU,false);
    extract_weights("data/conv4.txt",Layer4_Weights_CPU,false);
    extract_weights("data/conv5.txt",Layer5_Weights_CPU,false);
    extract_weights("data/fc6.txt",Layer6_Weights_CPU,false);
    extract_weights("data/fc7.txt",Layer7_Weights_CPU,false);
    extract_weights("data/fc8.txt",Layer8_Weights_CPU,false);
    printf("Extracted Weights and Bias successfully\n");
}
void Fill_bias(float *bias_1,float *bias_2,float *bias_3,float *bias_4,float *bias_5,float *bias_6,float *bias_7,float *bias_8)
{
    extract_weights("data/bias1.txt",bias_1,true);
    extract_weights("data/bias2.txt",bias_2,true);
    extract_weights("data/bias3.txt",bias_3,true);
    extract_weights("data/bias4.txt",bias_4,true);
    extract_weights("data/bias5.txt",bias_5,true);
    extract_weights("data/bias6.txt",bias_6,true);
    extract_weights("data/bias7.txt",bias_7,true);
    extract_weights("data/bias8.txt",bias_8,true);
}
void readIn(float *layer1)
{
    FILE *fp = fopen ("data/input.txt","rb");
    size_t len;
    char delim[1];
    delim[0] = '\n';
    int count = 0;
    char *token;
    char *line = NULL;
    if (fp != NULL)
    {
        printf(" File FOUND\n");
        {
            while ((getline(&line, &len, fp)) != -1)
            {
                token = strtok(line,delim);
                for(int n = 0; n < NUM; n++) {
                    layer1[n*INPUT_SIZE + count] = atof(token);
                }
                count++;		
            }
            /* printf("READ INPUT Final Count :: %d\n",count); */		
        }
        fclose(fp);
    }
    else
    {
        printf(" File NOt FOUND\n");
    }
}
void NeuralNetwork()
{
    cudaError_t err;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
#ifndef CPU// Set the CUDA device	
    int deviceCount;                                                         
    cudaGetDeviceCount(&deviceCount);                
    if (deviceCount == 0) {                                                  
        fprintf(stderr, "There is no device.\n");                            
        exit(EXIT_FAILURE);                                                  
    }                                                                        
    int dev;                                                                 
    for (dev = 0; dev < deviceCount; ++dev) {                                
        cudaDeviceProp deviceProp;                                           
        cudaGetDeviceProperties(&deviceProp, dev);   
        if (deviceProp.major >= 1)                                           
            break;                                                           
    }                                                                        
    if (dev == deviceCount) {                                                
        fprintf(stderr, "There is no device supporting CUDA.\n");            
        exit(EXIT_FAILURE);                                                  
    }                                                                        
    else                                                                     
        cudaSetDevice(dev);
#endif  
    /* Read Input File 227*227*3 */	
    float *Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));

        /* Declaration of Bias and Weights for CPU */ 
	float bias_1[96],bias_2[256],bias_3[384],bias_4[384],bias_5[256],bias_6[4096],bias_7[4096],bias_8[1000];
	float *Layer1_Weights_CPU = (float *)malloc(sizeof(float) *(L1_KERNEL_SIZE * L1_OUT));
	float *Layer2_Weights_CPU = (float *)malloc(sizeof(float) *(L2_KERNEL_SIZE * L2_OUT));
	float *Layer3_Weights_CPU = (float *)malloc(sizeof(float) *(L3_KERNEL_SIZE * L3_OUT));
	float *Layer4_Weights_CPU = (float *)malloc(sizeof(float) *(L4_KERNEL_SIZE * L4_OUT));
	float *Layer5_Weights_CPU = (float *)malloc(sizeof(float) *(L5_KERNEL_SIZE * L5_OUT));
	float *Layer6_Weights_CPU = (float *)malloc(sizeof(float) *(4096*256*6*6));
	float *Layer7_Weights_CPU = (float *)malloc(sizeof(float) *(4096*4096));
	float *Layer8_Weights_CPU = (float *)malloc(sizeof(float) *(4096*1000));
        
	float *Layer1_bias_GPU,*Layer1_Weights_GPU,*Layer1_Neurons_GPU,*Layer1_Norm_GPU,*Layer1_pool_GPU,*Layer2_Neurons_GPU;
	float *Layer2_bias_GPU,*Layer2_Weights_GPU,*Layer2_Norm_GPU,*Layer2_pool_GPU,*Layer3_Neurons_GPU;
	/* Third Layer convolution + ReLU  */ 
	float *Layer3_bias_GPU,*Layer3_Weights_GPU,*Layer4_Neurons_GPU;
	/* Fourth Layer convolution + ReLU  */
	float *Layer4_bias_GPU,*Layer4_Weights_GPU,*Layer5_Neurons_GPU;
	/* Fifth Layer convolution + ReLU + pooling */
	float *Layer5_bias_GPU,*Layer5_Weights_GPU,*Layer5_pool_GPU,*Layer6_Neurons_GPU;
	/* Sixth Layer Fully connected + ReLU */	
	float *Layer6_bias_GPU; 
	float *Layer6_Weights_GPU;
	float *Layer7_Neurons_GPU;
	/* Seventh Layer Fully connected + ReLU */	
	float *Layer7_bias_GPU; 
	float *Layer7_Weights_GPU;
	float *Layer8_Neurons_GPU;
	/* Eigth Layer Fully connected + ReLU */	
	float *Layer8_bias_GPU; 
	float *Layer9_Neurons_GPU;
	float *Layer8_Weights_GPU;

	cudaMallocManaged((void**) &Layer1_Weights_GPU, sizeof(float)* L1_KERNEL_SIZE * L1_OUT);
	cudaMallocManaged((void**) &Layer2_Weights_GPU,sizeof(float)*(L2_KERNEL_SIZE * L2_OUT));
	cudaMallocManaged((void**) &Layer3_Weights_GPU,sizeof(float)*(L3_KERNEL_SIZE * L3_OUT));
	cudaMallocManaged((void**) &Layer4_Weights_GPU,sizeof(float)*(L4_KERNEL_SIZE * L4_OUT));
	cudaMallocManaged((void**) &Layer5_Weights_GPU,sizeof(float)*(L5_KERNEL_SIZE * L5_OUT));
	cudaMallocManaged((void**) &Layer6_Weights_GPU,sizeof(float)*4096*256*6*6);
	cudaMallocManaged((void**) &Layer7_Weights_GPU,sizeof(float)*4096*4096);
	cudaMallocManaged((void**) &Layer8_Weights_GPU,sizeof(float)*4096*1000);

	cudaMallocManaged((void**) &Layer1_bias_GPU, sizeof(float)* L1_OUT * NUM);
	cudaMallocManaged((void**) &Layer2_bias_GPU, sizeof(float)* L2_OUT * NUM);
	cudaMallocManaged((void**) &Layer3_bias_GPU, sizeof(float)*L3_OUT * NUM);
	cudaMallocManaged((void**) &Layer4_bias_GPU, sizeof(float)*L4_OUT * NUM);
	cudaMallocManaged((void**) &Layer5_bias_GPU, sizeof(float)*L5_OUT * NUM);
	cudaMallocManaged((void**) &Layer6_bias_GPU, sizeof(float)*4096 * NUM);
	cudaMallocManaged((void**) &Layer7_bias_GPU, sizeof(float)*4096 * NUM);
	cudaMallocManaged((void**) &Layer8_bias_GPU, sizeof(float)*1000 * NUM);

	cudaMallocManaged((void**) &Layer1_Neurons_GPU, sizeof(float)* INPUT_SIZE * NUM);
	cudaMallocManaged((void**) &Layer1_Norm_GPU, sizeof(float)* (L1_OUT * L1_FMAP) * NUM);
	cudaMallocManaged((void**) &Layer1_pool_GPU,sizeof(float)* L1_OUT*L1_FMAP * NUM);
	cudaMallocManaged((void**) &Layer2_Neurons_GPU,sizeof(float)*L1_OUT * POOL1_FMAP * NUM);
	cudaMallocManaged((void**) &Layer2_Norm_GPU, sizeof(float)* L2_OUT * L2_FMAP * NUM);
	cudaMallocManaged((void**) &Layer2_pool_GPU,sizeof(float)*L2_OUT * L2_FMAP * NUM);
	cudaMallocManaged((void**) &Layer3_Neurons_GPU,sizeof(float)*L2_OUT * POOL2_FMAP * NUM);
	cudaMallocManaged((void**) &Layer4_Neurons_GPU, sizeof(float)*(L3_FMAP * L3_OUT) * NUM);
	cudaMallocManaged((void**) &Layer5_Neurons_GPU, sizeof(float)*(L4_FMAP * L4_OUT) * NUM);
	cudaMallocManaged((void**) &Layer5_pool_GPU, sizeof(float)*(L5_FMAP * L5_OUT) * NUM);
	cudaMallocManaged((void**) &Layer6_Neurons_GPU,sizeof(float)*L5_OUT * POOL3_FMAP * NUM);
	cudaMallocManaged((void**) &Layer7_Neurons_GPU, sizeof(float)*4096 * NUM);
	cudaMallocManaged((void**) &Layer8_Neurons_GPU, sizeof(float)*4096 * NUM);
	cudaMallocManaged((void**) &Layer9_Neurons_GPU, sizeof(float)*1000 * NUM);

	memset((char*) Layer1_Neurons_GPU, 0, sizeof(float)* INPUT_SIZE * NUM);
	memset((char*) Layer1_Norm_GPU, 0, sizeof(float)* (L1_OUT * L1_FMAP) * NUM);
	memset((char*) Layer1_pool_GPU,0, sizeof(float)* L1_OUT*L1_FMAP * NUM);
	memset((char*) Layer2_Neurons_GPU,0, sizeof(float)*L1_OUT * POOL1_FMAP * NUM);
	memset((char*) Layer2_Norm_GPU, 0, sizeof(float)* L2_OUT * L2_FMAP * NUM);
	memset((char*) Layer2_pool_GPU,0, sizeof(float)*L2_OUT * L2_FMAP * NUM);
	memset((char*) Layer3_Neurons_GPU,0, sizeof(float)*L2_OUT * POOL2_FMAP * NUM);
	memset((char*) Layer4_Neurons_GPU, 0, sizeof(float)*(L3_FMAP * L3_OUT) * NUM);
	memset((char*) Layer5_Neurons_GPU, 0, sizeof(float)*(L4_FMAP * L4_OUT) * NUM);
	memset((char*) Layer5_pool_GPU, 0, sizeof(float)*(L5_FMAP * L5_OUT) * NUM);
	memset((char*) Layer6_Neurons_GPU,0, sizeof(float)*L5_OUT * POOL3_FMAP * NUM);
	memset((char*) Layer7_Neurons_GPU, 0, sizeof(float)*4096 * NUM);
	memset((char*) Layer8_Neurons_GPU, 0, sizeof(float)*4096 * NUM);
	memset((char*) Layer9_Neurons_GPU, 0, sizeof(float)*1000 * NUM);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Mallocs failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    /* Fill Bias and Weights */	
    Fill_bias(Layer1_bias_GPU,Layer2_bias_GPU,Layer3_bias_GPU,Layer4_bias_GPU,Layer5_bias_GPU,Layer6_bias_GPU,Layer7_bias_GPU,Layer8_bias_GPU);
    Fill_weights(Layer1_Weights_GPU,Layer2_Weights_GPU,Layer3_Weights_GPU,Layer4_Weights_GPU,Layer5_Weights_GPU,Layer6_Weights_GPU,Layer7_Weights_GPU,Layer8_Weights_GPU);
	readIn(Layer1_Neurons_GPU);

    /* Output is 96*55*55 , hence launch as 96*32*32 + 96*23*23 */
	dim3 Layer1_Block(96*NUM,1,1);
	dim3 Layer1_Thread(32,32);	
	dim3 Layer11_Block(96*NUM,1,1);
	dim3 Layer11_Thread(32,23);	
	dim3 Layer12_Block(96*NUM,1,1);
	dim3 Layer12_Thread(23,32);	
	dim3 Layer13_Block(96*NUM,1,1);
	dim3 Layer13_Thread(23,23);	
	dim3 Norm1_Block(96*NUM,1,1);
	dim3 Norm1_Thread(32,32);   
	dim3 Norm11_Block(96*NUM,1,1);
	dim3 Norm11_Thread(32,23);   
	dim3 Norm12_Block(96*NUM,1,1);
	dim3 Norm12_Thread(23,32);   
	dim3 Norm13_Block(96*NUM,1,1);
	dim3 Norm13_Thread(23,23);   
	dim3 pool1_Block(96*NUM,1,1);
	dim3 pool1_Thread(27,27);   
	dim3 Layer2_Block(128*NUM,1,1);
	dim3 Layer2_Thread(27,27);   
	dim3 Norm2_Block(256*NUM,1,1);
	dim3 Norm2_Thread(27,27);   
	dim3 pool2_Block(256*NUM,1,1);
	dim3 pool2_Thread(13,13);   
	dim3 Layer3_Block(384*NUM,1,1);
	dim3 Layer3_Thread(13,13);   
	dim3 Layer4_Block(192*NUM,1,1);
	dim3 Layer4_Thread(13,13);   
	dim3 Layer5_Block(128*NUM,1,1);
	dim3 Layer5_Thread(13,13);   
	dim3 pool5_Block(256*NUM,1,1);
	dim3 pool5_Thread(6,6);   
	dim3 Layer6_Block(4096*NUM,1,1);
	dim3 Layer6_Thread(1,1);   // combi tried 10*10*10
	dim3 Layer7_Block(4096*NUM,1,1);
	dim3 Layer7_Thread(1,1);   // combi tried 10*10*10
	dim3 Layer8_Block(1000*NUM,1,1);
	dim3 Layer8_Thread(1,1);   // combi tried 10*10*10

    nvml_start();
    penguinStartStatCollection();
    begin = std::chrono::steady_clock::now();

    cudaDeviceSynchronize();
    executeFirstLayer1<<<Layer1_Block,Layer1_Thread>>>(Layer1_bias_GPU,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer1_Norm_GPU,0,0, 96, INPUT_SIZE, L1_OUT*L1_FMAP);
	executeFirstLayer2<<<Layer11_Block,Layer11_Thread>>>(Layer1_bias_GPU,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer1_Norm_GPU,0,32, 96, INPUT_SIZE, L1_OUT*L1_FMAP);
	executeFirstLayer3<<<Layer12_Block,Layer12_Thread>>>(Layer1_bias_GPU,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer1_Norm_GPU,32,0, 96, INPUT_SIZE, L1_OUT*L1_FMAP);
	executeFirstLayer4<<<Layer13_Block,Layer13_Thread>>>(Layer1_bias_GPU,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer1_Norm_GPU,32,32, 96, INPUT_SIZE, L1_OUT*L1_FMAP);

	executelrnNormCuda_split1<<<Norm1_Block,Norm1_Thread>>>(Layer1_Norm_GPU,0.0001,0.75,5,96,55,55,Layer1_pool_GPU,0,0, 96, L1_OUT*L1_FMAP, L1_OUT*L1_FMAP);
	executelrnNormCuda_split2<<<Norm11_Block,Norm11_Thread>>>(Layer1_Norm_GPU,0.0001,0.75,5,96,55,55,Layer1_pool_GPU,0,32, 96, L1_OUT*L1_FMAP, L1_OUT*L1_FMAP);
	executelrnNormCuda_split3<<<Norm12_Block,Norm12_Thread>>>(Layer1_Norm_GPU,0.0001,0.75,5,96,55,55,Layer1_pool_GPU,32,0, 96, L1_OUT*L1_FMAP, L1_OUT*L1_FMAP);
	executelrnNormCuda_split4<<<Norm13_Block,Norm13_Thread>>>(Layer1_Norm_GPU,0.0001,0.75,5,96,55,55,Layer1_pool_GPU,32,32, 96, L1_OUT*L1_FMAP, L1_OUT*L1_FMAP);

	executepoolingCuda1<<<pool1_Block,pool1_Thread>>>(Layer1_pool_GPU,Layer2_Neurons_GPU,96,27,27,3,2,55,55, 96,L1_OUT*L1_FMAP, L1_OUT*POOL1_FMAP);

	execute3DconvolutionCuda1<<<Layer2_Block,Layer2_Thread>>>(Layer2_bias_GPU,Layer2_Neurons_GPU,Layer2_Weights_GPU,Layer2_Norm_GPU,128,27,27,1,5,2,48,2, 128, L1_OUT*POOL1_FMAP, L2_OUT*L2_FMAP);
	execute3Dconvolutiongroup2Cuda1<<<Layer2_Block,Layer2_Thread>>>(Layer2_bias_GPU,Layer2_Neurons_GPU,Layer2_Weights_GPU,Layer2_Norm_GPU,128,27,27,1,5,2,48,2, 128, L1_OUT*POOL1_FMAP, L2_OUT*L2_FMAP);

	executelrnNormCuda<<<Norm2_Block,Norm2_Thread>>>(Layer2_Norm_GPU,0.0001,0.75,5,256,27,27,Layer2_pool_GPU,0, 256, L2_OUT*L2_FMAP, L2_OUT*L2_FMAP);
	executepoolingCuda2<<<pool2_Block,pool2_Thread>>>(Layer2_pool_GPU,Layer3_Neurons_GPU,256,13,13,3,2,27,27, 256, L2_OUT*L2_FMAP, L2_OUT*POOL2_FMAP);

	execute3DconvolutionCuda2<<<Layer3_Block,Layer3_Thread>>>(Layer3_bias_GPU,Layer3_Neurons_GPU,Layer3_Weights_GPU,Layer4_Neurons_GPU,384,13,13,1,3,1,256,1, 384, L2_OUT*POOL2_FMAP, L3_FMAP*L3_OUT);
	execute3DconvolutionCuda3<<<Layer4_Block,Layer4_Thread>>>(Layer4_bias_GPU,Layer4_Neurons_GPU,Layer4_Weights_GPU,Layer5_Neurons_GPU,192,13,13,1,3,1,192,2, 192, L3_FMAP*L3_OUT, L4_FMAP*L4_OUT);

	execute3Dconvolutiongroup2Cuda2<<<Layer4_Block,Layer4_Thread>>>(Layer4_bias_GPU,Layer4_Neurons_GPU,Layer4_Weights_GPU,Layer5_Neurons_GPU,192,13,13,1,3,1,192,2, 192, L3_FMAP*L3_OUT, L4_FMAP*L4_OUT);


    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    std::cout << "GPU.Parser.Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
    nvml_stop();
    penguinStopStatCollection();
    return;
    // Program buggy after this.
    // TODO: Fix and run

	/* execute3DconvolutionCuda<<<Layer5_Block,Layer5_Thread>>>(Layer5_bias_GPU,Layer5_Neurons_GPU,Layer5_Weights_GPU,Layer5_pool_GPU,128,13,13,1,3,1,192,2, 128, L4_FMAP*L4_OUT, L5_FMAP*L5_OUT); */
	/* execute3Dconvolutiongroup2Cuda<<<Layer5_Block,Layer5_Thread>>>(Layer5_bias_GPU,Layer5_Neurons_GPU,Layer5_Weights_GPU,Layer5_pool_GPU,128,13,13,1,3,1,192,2, 128, L4_FMAP*L4_OUT, L5_FMAP*L5_OUT ); */

	/* executepoolingCuda3<<<pool5_Block,pool5_Thread>>>(Layer5_pool_GPU,Layer6_Neurons_GPU,256,6,6,3,2,13,13, 256, L5_OUT*L5_FMAP, L5_OUT*POOL3_FMAP); */

	/* executeFCLayer<<<Layer6_Block,Layer6_Thread>>>(Layer6_bias_GPU,Layer6_Neurons_GPU,Layer6_Weights_GPU,Layer7_Neurons_GPU,4096,(256*6*6),true,false, 4096, L5_OUT*POOL3_FMAP, 4096); */
	/* // RELU LAyer */ 

	/* executeFCLayer<<<Layer7_Block,Layer7_Thread>>>(Layer7_bias_GPU,Layer7_Neurons_GPU,Layer7_Weights_GPU,Layer8_Neurons_GPU,4096,4096,true,false, 4096, 4096, 4096); */

	/* executeFCLayer<<<Layer8_Block,Layer8_Thread>>>(Layer8_bias_GPU,Layer8_Neurons_GPU,Layer8_Weights_GPU,Layer9_Neurons_GPU,1000,4096,false,false, 1000, 4096, 1000); */


    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Layer1 failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    /* return; */

    /* debuggy(Layer9_Neurons_GPU, 1000); */
    for(int n =0; n < NUM; n++) {
	float max = 0.0;int index = 0; 
        for(int i =0; i < 1000; i++)
        {
            if(max < Layer9_Neurons_GPU[n*1000+i])
            {
                max = Layer9_Neurons_GPU[n*1000+i];
                index = i;
            }
        }
        /* printf("INDEX = %d\n",index); */
    }
    cudaFree(Layer1_Neurons_GPU);
    cudaFree(Layer1_Weights_GPU);
    cudaFree(Layer1_bias_GPU);
    cudaFree(Layer1_Norm_GPU);
    cudaFree(Layer1_pool_GPU);
    cudaFree(Layer2_Neurons_GPU);
    cudaFree(Layer2_Weights_GPU);
    cudaFree(Layer2_bias_GPU);
    cudaFree(Layer2_pool_GPU);
    cudaFree(Layer2_Norm_GPU);
    cudaFree(Layer3_Neurons_GPU);
    cudaFree(Layer3_Weights_GPU);
    cudaFree(Layer3_bias_GPU);
    cudaFree(Layer4_Neurons_GPU);
    cudaFree(Layer4_Weights_GPU);
    cudaFree(Layer4_bias_GPU);
    cudaFree(Layer5_Neurons_GPU);
    cudaFree(Layer5_Weights_GPU);
    cudaFree(Layer5_bias_GPU);
    cudaFree(Layer5_pool_GPU);
    cudaFree(Layer6_Neurons_GPU);
    cudaFree(Layer6_Weights_GPU);
    cudaFree(Layer6_bias_GPU);
    cudaFree(Layer7_Neurons_GPU);
    cudaFree(Layer7_bias_GPU);
    cudaFree(Layer7_Weights_GPU);
    cudaFree(Layer8_Neurons_GPU);
    cudaFree(Layer8_Weights_GPU);
    cudaFree(Layer8_bias_GPU);
    cudaFree(Layer9_Neurons_GPU);
    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer2_Weights_CPU);
    free(Layer3_Weights_CPU);
    free(Layer4_Weights_CPU);
    free(Layer5_Weights_CPU);
    free(Layer6_Weights_CPU);
    free(Layer7_Weights_CPU);
    free(Layer8_Weights_CPU);
	/* SoftMax */
	//Confirm the functionality of SoftMax ,extract_weights("data/fc8_out.txt",fc9_Neurons_CPU,false);
	//executeSoftMax(fc9_Neurons_CPU);
	exit(0);
}


			
