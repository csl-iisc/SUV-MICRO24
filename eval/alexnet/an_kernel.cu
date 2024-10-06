/**********************************************************************
* FILENAME :        an_kernel.cu             
* 
* DESCRIPTION :
*       Kernel side implementation of AlexNet network
*
* NOTES :
*       This file includes implementation of 2D/3D convolution
*       normalisation,pooling and fully connected layer kernels.
* 
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/
#ifndef _AN_KERNEL_H_
#define _AN_KERNEL_H_

#include <stdio.h>

#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif


//#define LAYER2_DEBUG 
//#define POOL_DEBUG 
__constant__ int kernelTemplate[25] = {
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };
__constant__ int kernelTemplate2[25] = {
        0,  1,  2,  3,  4,
        13, 14, 15, 16, 17, 
        26, 27, 28, 29, 30,
        39, 40, 41, 42, 43, 
        52, 53, 54, 55, 56   };

__global__ void executeFirstLayer1(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int r_offset, int c_offset, int bsize, int bsize_l1ng, int bsize_l2ng)
{
    float product = 0.0;
    int col_width = 227;
    int stride_width = 4;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x + r_offset;
    int col = threadIdx.y + c_offset;
    colstride = 3*row*stride_width*col_width;
    stride = 0;
    product = 0;
    stride = col * 4 * 3;
    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            product +=        ((Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
        }
    }
    product += bias[output];
    if(product < 0) /* RELU Layer */
        product = 0; // max(0,x)
    Layer2_Neurons_GPU[batchid*bsize_l2ng + output*55*55 + row*55 + col] = product;
    product = 0.0;
}
__global__ void executeFirstLayer2(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int r_offset, int c_offset, int bsize, int bsize_l1ng, int bsize_l2ng)
{
    float product = 0.0;
    int col_width = 227;
    int stride_width = 4;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x + r_offset;
    int col = threadIdx.y + c_offset;
    colstride = 3*row*stride_width*col_width;
    stride = 0;
    product = 0;
    stride = col * 4 * 3;
    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            product +=        ((Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
        }
    }
    product += bias[output];
    if(product < 0) /* RELU Layer */
        product = 0; // max(0,x)
    Layer2_Neurons_GPU[batchid*bsize_l2ng + output*55*55 + row*55 + col] = product;
    product = 0.0;
}
__global__ void executeFirstLayer3(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int r_offset, int c_offset, int bsize, int bsize_l1ng, int bsize_l2ng)
{
    float product = 0.0;
    int col_width = 227;
    int stride_width = 4;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x + r_offset;
    int col = threadIdx.y + c_offset;
    colstride = 3*row*stride_width*col_width;
    stride = 0;
    product = 0;
    stride = col * 4 * 3;
    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            product +=        ((Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
        }
    }
    product += bias[output];
    if(product < 0) /* RELU Layer */
        product = 0; // max(0,x)
    Layer2_Neurons_GPU[batchid*bsize_l2ng + output*55*55 + row*55 + col] = product;
    product = 0.0;
}
__global__ void executeFirstLayer4(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int r_offset, int c_offset, int bsize, int bsize_l1ng, int bsize_l2ng)
{
    float product = 0.0;
    int col_width = 227;
    int stride_width = 4;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x + r_offset;
    int col = threadIdx.y + c_offset;
    colstride = 3*row*stride_width*col_width;
    stride = 0;
    product = 0;
    stride = col * 4 * 3;
    /* RGB weights and input 11*11*3 */
    for(int i = 0; i < 11; i++)
    {
        for(int j = 0; j < 11; j++)
        {
            product +=        ((Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*11 + j + (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11 + j+ (output * 11*11*3)])
                    + (Layer1_Neurons_GPU[batchid*bsize_l1ng + i*227*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU[i*11 + 11*11*2 + j+ (output * 11*11*3)]));
        }
    }
    product += bias[output];
    if(product < 0) /* RELU Layer */
        product = 0; // max(0,x)
    Layer2_Neurons_GPU[batchid*bsize_l2ng + output*55*55 + row*55 + col] = product;
    product = 0.0;
}

/* IN : Layer2_Neurons_GPU // Neurons input
        Layer2_pool_GPU    // output after pooling
        out                // number of outputs 
        out_fr             // feature map size of output in terms of row 
        out_fc             // feature map size of output in terms of column
        kernel             // kernel size
        stride_width       // stride
        in_fr             // feature map size of input in terms of row
        in_fc             // feature map size of input in terms of column 
*/
__global__ void executepoolingCuda1(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc, int bsize, int bsize_l2ng, int bsize_l2pg)
{
    float max = 0.0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x;
    int col = threadIdx.y;
    colstride = row * stride_width*in_fc;
    stride = col * stride_width;
    for(int i = 0; i < kernel; i++)
    {
        for(int j = 0; j < kernel; j++)
        {
            if(max < ((Layer2_Neurons_GPU[batchid*bsize_l2ng + (output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
                max =   ((Layer2_Neurons_GPU[batchid*bsize_l2ng + (output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;

        }
    }
    Layer2_pool_GPU[batchid*bsize_l2pg + output*out_fr*out_fc + row*out_fc + col] = max;
    max = 0.0;
    stride+= stride_width;
}
__global__ void executepoolingCuda2(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc, int bsize, int bsize_l2ng, int bsize_l2pg)
{
    float max = 0.0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int row = threadIdx.x;
    int col = threadIdx.y;
    colstride = row * stride_width*in_fc;
    stride = col * stride_width;
    for(int i = 0; i < kernel; i++)
    {
        for(int j = 0; j < kernel; j++)
        {
            if(max < ((Layer2_Neurons_GPU[batchid*bsize_l2ng + (output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
                max =   ((Layer2_Neurons_GPU[batchid*bsize_l2ng + (output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;

        }
    }
    Layer2_pool_GPU[batchid*bsize_l2pg + output*out_fr*out_fc + row*out_fc + col] = max;
    max = 0.0;
    stride+= stride_width;
}

__global__ void execute3DconvolutionCuda1(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group, int bsize, int bsize_l2ng, int bsize_l3ng)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize; // 128
    int batchid = blockIdx.x / bsize;
    colstride = 0;
    int row = threadIdx.x;
    stride = 0;
    /* if(row > pad)gcc */
       colstride = (row - pad) * fr;
    int col = threadIdx.y;
    /* if(col >= pad) */
        stride = (col - pad) * stride_width;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                if(((row - i) >= pad) && ((col - j) >= pad) && ((row + i) <= fr) && ((col + i) <= fc)) {
                    product += ( Layer2_Neurons_GPU[batchid*bsize_l2ng + feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
                }
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[batchid*bsize_l3ng + output*fr*fc + row*fc + col] = product;
    product = 0.0;
    if(col >= pad)
        stride+=stride_width;
}
__global__ void execute3DconvolutionCuda2(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group, int bsize, int bsize_l2ng, int bsize_l3ng)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize; // 128
    int batchid = blockIdx.x / bsize;
    colstride = 0;
    int row = threadIdx.x;
    stride = 0;
    /* if(row > pad)gcc */
       colstride = (row - pad) * fr;
    int col = threadIdx.y;
    /* if(col >= pad) */
        stride = (col - pad) * stride_width;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                if(((row - i) >= pad) && ((col - j) >= pad) && ((row + i) <= fr) && ((col + i) <= fc)) {
                    product += ( Layer2_Neurons_GPU[batchid*bsize_l2ng + feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
                }
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[batchid*bsize_l3ng + output*fr*fc + row*fc + col] = product;
    product = 0.0;
    if(col >= pad)
        stride+=stride_width;
}
__global__ void execute3DconvolutionCuda3(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group, int bsize, int bsize_l2ng, int bsize_l3ng)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x % bsize; // 128
    int batchid = blockIdx.x / bsize;
    colstride = 0;
    int row = threadIdx.x;
    stride = 0;
    /* if(row > pad)gcc */
       colstride = (row - pad) * fr;
    int col = threadIdx.y;
    /* if(col >= pad) */
        stride = (col - pad) * stride_width;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    for(int feature =0; feature < in_output ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                if(((row - i) >= pad) && ((col - j) >= pad) && ((row + i) <= fr) && ((col + i) <= fc)) {
                    product += ( Layer2_Neurons_GPU[batchid*bsize_l2ng + feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + feature*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]);
                }
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[batchid*bsize_l3ng + output*fr*fc + row*fc + col] = product;
    product = 0.0;
    if(col >= pad)
        stride+=stride_width;
}

__global__ void execute3Dconvolutiongroup2Cuda1(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group, int bsize, int bsize_l2ng, int bsize_l3ng)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    /* Execute second set of inputs */
    int output = (blockIdx.x % bsize) + out;
    int batchid = blockIdx.x / bsize;
    colstride = 0;
    int row = threadIdx.x;
    stride = 0;
    /* if(row > pad) */
        colstride = (row - pad) * fr;
    int col = threadIdx.y;
    /* if(col >= pad) */
        stride = (col-pad) *stride_width;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    /* take care of padding in left hand side of image*/
    for(int feature = in_output ; feature < (in_output << 1) ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                if(((row - i) >= pad) && ((col - j) >= pad) && ((row + i) <= fr) && ((col + i) <= fc)) {
                    product += (( Layer2_Neurons_GPU[batchid*bsize_l2ng + feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + (feature-in_output)*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]));
                }
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[batchid*bsize_l3ng + output*fr*fc + row*fc + col] = product;
    product = 0.0;
}
__global__ void execute3Dconvolutiongroup2Cuda2(float *bias,float *Layer2_Neurons_GPU, float *Layer2_Weights_GPU,float *Layer3_Neurons_GPU,int out,int fr,int fc,int stride_width,int kernel,int pad,int in_output,int group, int bsize, int bsize_l2ng, int bsize_l3ng)
{
    float product = 0.0;
    int x_pad = 0, y_pad = 0, loopc = 0,loopr = 0;
    int stride = 0,colstride = 0;
    /* Execute second set of inputs */
    int output = (blockIdx.x % bsize) + out;
    int batchid = blockIdx.x / bsize;
    colstride = 0;
    int row = threadIdx.x;
    stride = 0;
    /* if(row > pad) */
        colstride = (row - pad) * fr;
    int col = threadIdx.y;
    /* if(col >= pad) */
        stride = (col-pad) *stride_width;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    /* take care of padding in left hand side of image*/
    for(int feature = in_output ; feature < (in_output << 1) ; feature++) // calculate the feature maps
    {
        for(int i =0; i < loopr ; i++) // kernel convolution
        {
            for(int j =0; j < loopc ; j++) // kernel convolution
            {
                if(((row - i) >= pad) && ((col - j) >= pad) && ((row + i) <= fr) && ((col + i) <= fc)) {
                    product += (( Layer2_Neurons_GPU[batchid*bsize_l2ng + feature*fr*fc + i*fc + j + stride + colstride] * Layer2_Weights_GPU[output*kernel*kernel*in_output + (feature-in_output)*kernel*kernel + i*kernel + j + kernel*x_pad + y_pad]));
                }
            }
        }
    }
    product += bias[output];
    if(product < 0) /* ReLU Layer */
        product = 0;
    Layer3_Neurons_GPU[batchid*bsize_l3ng + output*fr*fc + row*fc + col] = product;
    product = 0.0;
}

__global__ void executelrnNormCuda_split1(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU,int r_offset, int c_offset, int bsize, int bsize_ling, int bsize_long)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = blockIdx.x % bsize;
        int batchid = blockIdx.x / bsize;
        int row = threadIdx.x + r_offset;
        int col = threadIdx.y + c_offset;
        /* nStart=(output-2) > 1 ? (output-2) : 1 ; */
        /* nEnd=(output+2) <  out ? (output+2) : out ; */
        /* for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution */
        /* { */
        /*     sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + i*fr*fc + row*fc + col]),2); */
        /* } */
        // PRatheek: fixed logic, I hope
        for(int i = 0; i < 4; i++) // kernel convolution
        {
            int index = output - 3 + i;
            if((index >= 0) && (index < out)) {
                sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + output+i*fr*fc + row*fc + col]),2);
            }
        }
        value = (Layer_InNeurons_GPU[batchid*bsize_ling + output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
        sum = 0;
        Layer_OutNeurons_GPU[batchid*bsize_long + output*fr*fc + row*fc + col] = value;
}
__global__ void executelrnNormCuda_split2(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU,int r_offset, int c_offset, int bsize, int bsize_ling, int bsize_long)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = blockIdx.x % bsize;
        int batchid = blockIdx.x / bsize;
        int row = threadIdx.x + r_offset;
        int col = threadIdx.y + c_offset;
        /* nStart=(output-2) > 1 ? (output-2) : 1 ; */
        /* nEnd=(output+2) <  out ? (output+2) : out ; */
        /* for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution */
        /* { */
        /*     sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + i*fr*fc + row*fc + col]),2); */
        /* } */
        // PRatheek: fixed logic, I hope
        for(int i = 0; i < 4; i++) // kernel convolution
        {
            int index = output - 3 + i;
            if((index >= 0) && (index < out)) {
                sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + output+i*fr*fc + row*fc + col]),2);
            }
        }
        value = (Layer_InNeurons_GPU[batchid*bsize_ling + output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
        sum = 0;
        Layer_OutNeurons_GPU[batchid*bsize_long + output*fr*fc + row*fc + col] = value;
}
__global__ void executelrnNormCuda_split3(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU,int r_offset, int c_offset, int bsize, int bsize_ling, int bsize_long)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = blockIdx.x % bsize;
        int batchid = blockIdx.x / bsize;
        int row = threadIdx.x + r_offset;
        int col = threadIdx.y + c_offset;
        /* nStart=(output-2) > 1 ? (output-2) : 1 ; */
        /* nEnd=(output+2) <  out ? (output+2) : out ; */
        /* for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution */
        /* { */
        /*     sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + i*fr*fc + row*fc + col]),2); */
        /* } */
        // PRatheek: fixed logic, I hope
        for(int i = 0; i < 4; i++) // kernel convolution
        {
            int index = output - 3 + i;
            if((index >= 0) && (index < out)) {
                sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + output+i*fr*fc + row*fc + col]),2);
            }
        }
        value = (Layer_InNeurons_GPU[batchid*bsize_ling + output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
        sum = 0;
        Layer_OutNeurons_GPU[batchid*bsize_long + output*fr*fc + row*fc + col] = value;
}
__global__ void executelrnNormCuda_split4(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU,int r_offset, int c_offset, int bsize, int bsize_ling, int bsize_long)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = blockIdx.x % bsize;
        int batchid = blockIdx.x / bsize;
        int row = threadIdx.x + r_offset;
        int col = threadIdx.y + c_offset;
        /* nStart=(output-2) > 1 ? (output-2) : 1 ; */
        /* nEnd=(output+2) <  out ? (output+2) : out ; */
        /* for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution */
        /* { */
        /*     sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + i*fr*fc + row*fc + col]),2); */
        /* } */
        // PRatheek: fixed logic, I hope
        for(int i = 0; i < 4; i++) // kernel convolution
        {
            int index = output - 3 + i;
            if((index >= 0) && (index < out)) {
                sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + output+i*fr*fc + row*fc + col]),2);
            }
        }
        value = (Layer_InNeurons_GPU[batchid*bsize_ling + output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
        sum = 0;
        Layer_OutNeurons_GPU[batchid*bsize_long + output*fr*fc + row*fc + col] = value;
}

__global__ void executelrnNormCuda(float *Layer_InNeurons_GPU, float alpha, float beta,int local_size,int out,int fr,int fc,float *Layer_OutNeurons_GPU,int func_call, int bsize, int bsize_ling, int bsize_long)
{
        int nStart = 0, nEnd = 0;
        float value = 0.0;float sum = 0.0;
        int output = blockIdx.x % bsize;
        int batchid = blockIdx.x / bsize;
        int row = threadIdx.x + func_call * 32;
        int col = threadIdx.y + func_call * 32;
        /* nStart=(output-2) > 1 ? (output-2) : 1 ; */
        /* nEnd=(output+2) <  out ? (output+2) : out ; */
        /* for(int i = (nStart-1); i < (nEnd-1) ; i++) // kernel convolution */
        /* { */
        /*     sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + i*fr*fc + row*fc + col]),2); */
        /* } */
        for(int i = 0; i < 4; i++) // kernel convolution
        {
            int index = output - 3 + i;
            if((index >= 0) && (index < out)) {
                sum += pow(( Layer_InNeurons_GPU[batchid*bsize_ling + output+i*fr*fc + row*fc + col]),2);
            }
        }
        value = (Layer_InNeurons_GPU[batchid*bsize_ling + output*fr*fc + row*fc + col]) / (pow( 1 + ((alpha/local_size) *sum),beta));
        sum = 0;
        Layer_OutNeurons_GPU[batchid*bsize_long + output*fr*fc + row*fc + col] = value;
}

__global__ void executeFCLayer(float *bias,float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int output, int input,bool reLU,bool dropout, int bsize, int bsize_ling, int bsize_long)
{
    float product = 0.0;
    int out = blockIdx.x % bsize;
    int batchid = blockIdx.x / bsize;
    int weight = out * input;
    {
        for(int in = 0; in < input; in++)
        {
               product += Layer_InNeurons_GPU[batchid * bsize_ling + in] * Layer_Weights_GPU[weight+in];
        }
        product += bias[out];
        if(reLU == true)
        {
            if(product < 0) /* ReLU Layer */
                product = 0;
        }

        Layer_OutNeurons_GPU[batchid * bsize_long + out] = product;
        product = 0.0;
    }
}

__global__ void executeThirdLayer(float *Layer3_Neurons_GPU, float *Layer3_Weights_GPU,float *Layer4_Neurons_GPU)
{
    int blockID=blockIdx.x;
    //int pixelY=threadIdx.y;


    int weightBegin=blockID*1251;

    float result=0;

    result+=Layer3_Weights_GPU[weightBegin];

    ++weightBegin;

    for (int i=0; i<1250; ++i )
    {
        result+=Layer3_Neurons_GPU[i+(1250*blockIdx.y)]*Layer3_Weights_GPU[weightBegin+i];
    }

    result=(1.7159*tanhf(0.66666667*result));

    Layer4_Neurons_GPU[blockID+(100*blockIdx.y)]=result;

}

__global__ void executeFourthLayer(float *Layer4_Neurons_GPU,float *Layer4_Weights_GPU,float *Layer5_Neurons_GPU)
{
    int blockID=blockIdx.x;
    //int pixelY=threadIdx.y;


    int weightBegin=blockID*101;

    float result=0;

    result+=Layer4_Weights_GPU[weightBegin];

    ++weightBegin;

    for (int i=0; i<100; ++i )
    {
        result+=Layer4_Neurons_GPU[i+(100*blockIdx.y)]*Layer4_Weights_GPU[weightBegin+i];
    }

    result=(1.7159*tanhf(0.66666667*result));

    Layer5_Neurons_GPU[blockID+(10*blockIdx.y)]=result;
}


#endif // #ifndef _AN_KERNEL_H_
