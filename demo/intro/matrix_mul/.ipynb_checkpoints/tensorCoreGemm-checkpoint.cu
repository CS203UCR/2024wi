/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include <pthread.h>
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}



__global__ void print_gpu_values(half *a, half *b, half *c_half, float *c_float)
{
    printf("a %f, b %f, c_half %f, c_float %f\n", __half2float(*a), __half2float(*b), __half2float(*c_half), *c_float);
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n, float scale) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      if(scale > 32768.0)
          out[idx] = (half)(in[idx]/scale);
      else
      out[idx] = in[idx];
   }
}


int main(int argc, char* argv[]) {
   float *a_fp32, *A;
   float *b_fp32, *B;
   float scale;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas_gemmEx;

   float *c_host_cublas;
   float *c_host_cublasCublasGemmEx;


   float alpha = 1.0f;
   float beta = 0.0f;
   int mode = 0,i,ret,current;
   FILE *fin;

   int MATRIX_M=10240;
   int MATRIX_N=10240;
   int MATRIX_K=10240;
   
   cublasHandle_t cublasHandle;
   

   
   cudaEvent_t startcublasEX;
   cudaEvent_t stopcublasEX;



   cudaEvent_t startcublasCublasGemmEx;
   cudaEvent_t stopcublasCublasGemmEx;
   
   struct timeval time_start, time_end, total_start, total_end;
   int init_time;
   gettimeofday(&time_start, NULL);
   cudaFree(0);

   cudaErrCheck(cudaEventCreate(&startcublasEX));
   cudaErrCheck(cudaEventCreate(&stopcublasEX));

   cudaErrCheck(cudaEventCreate(&startcublasCublasGemmEx));
   cudaErrCheck(cudaEventCreate(&stopcublasCublasGemmEx));
   A = (float*)malloc(MATRIX_M * MATRIX_K * sizeof(float));
   B = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));
    if(mode == 0)
    {
        fin = fopen(argv[1], "r");
        for (i = 0; i < (MATRIX_M*MATRIX_K); i++) {
            ret = fscanf(fin,"%f",&A[i]); 
        }
        fclose(fin);
        fin = fopen(argv[2], "r");
        for (i = 0; i < (MATRIX_K * MATRIX_N); i++) {
            ret = fscanf(fin,"%f",&B[i]);
        }
        fclose(fin);
    }
    else
    {
        fin = fopen(argv[1], "rb");
        for (i = 0; i < (MATRIX_M*MATRIX_K); i++) {
            ret = fread(&current, 1, sizeof(int), fin);
            A[i] = (float)current;
        }
        fclose(fin);
        fin = fopen(argv[2], "rb");
        for (i = 0; i < (MATRIX_K * MATRIX_N); i++) {
            ret = fread(&current, 1, sizeof(int), fin);
            B[i] = (float)current;
        }
        fclose(fin);
    }

   gettimeofday(&total_start, NULL);

   cublasErrCheck(cublasCreate(&cublasHandle));

   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  	gettimeofday(&total_end, NULL);
        init_time = ((total_end.tv_sec * 1000000 + total_end.tv_usec) - (total_start.tv_sec * 1000000 + total_start.tv_usec));
	fprintf(stderr,"cublasCreate(2)  %d\n",init_time);
   MATRIX_M = atoi(argv[3]);
   MATRIX_K = MATRIX_M;
   MATRIX_N = MATRIX_M;
   cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));

   cudaErrCheck(cudaMalloc((void**)&c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_cublasCublasGemmEx = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   cudaErrCheck(cudaMemcpy(a_fp32, A, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp32, B, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice)); 
   scale=3.4028234664e+38;
   
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K, scale);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N, scale);

   cudaErrCheck(cudaMemcpy(c_cublas_gemmEx, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

  	gettimeofday(&time_end, NULL);
        init_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	fprintf(stderr,"Before GEMM %d\n",init_time);

   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // Now using cuBLAS but not tensor
   printf("Running with cuBLAS (GemmEX)...\n");
   cudaErrCheck(cudaEventRecord(startcublasCublasGemmEx));

   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas_gemmEx, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
   cudaErrCheck(cudaEventRecord(stopcublasCublasGemmEx));

   // Error checking
   cudaErrCheck(cudaMemcpy(c_host_cublasCublasGemmEx, c_cublas_gemmEx, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));


      float cublasTime;

      cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublasCublasGemmEx, stopcublasCublasGemmEx));
      printf("cublas cublasGemmEx took %f ms, GFLOS: %lf\n", cublasTime, (double)((double)MATRIX_M * (double)MATRIX_N*(double)MATRIX_K)*2/((double)cublasTime*1000000));


   cudaErrCheck(cudaEventDestroy(startcublasCublasGemmEx));
   cudaErrCheck(cudaEventDestroy(stopcublasCublasGemmEx));


   cudaErrCheck(cudaFree(a_fp32));
   cudaErrCheck(cudaFree(b_fp32));
   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas_gemmEx));
   
   free(c_host_cublas);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}


