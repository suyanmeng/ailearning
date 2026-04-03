#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

__global__ void vecAddGPU(const float* A, const float* B,float* C,int N)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N)
    {
        C[idx] = A[idx]+B[idx];
    }
}

void vecAddCpu(const float* A, const float* B,float* C,int N){
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i]+B[i];
    }
    
}

int main ()
{
    int N = 1024* 1024 *100;
    int size = N*sizeof(float);
    float *h_A= new float[N];
    float *h_B= new float[N];
    float *h_C= new float[N]{};

    for (int i = 0; i < N; i++)
    {
        h_A[i]=1.0f;
        h_B[i]=2.0f;
    }
    auto t1=chrono::high_resolution_clock::now();
    vecAddCpu(h_A,h_B,h_C,N);
    auto t2=chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double>(t2-t1).count()*1000;
    cout << "CPU 单核耗时: " << cpu_time << " ms" << endl;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N+blockSize-1)/blockSize;
    auto t3 = chrono::high_resolution_clock::now();
    vecAddGPU<<<gridSize,blockSize>>>(d_A,d_B,d_C,N);//GPU 核启动有固定开销：10~20µs,干少量的活比cpu慢
    cudaDeviceSynchronize();
    auto t4 = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double>(t4-t3).count()*1000;
    cout << "gpu耗时: " << gpu_time << " ms" << endl;
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    bool flag = true;
    for(int i = 0;i<N;i++)
    {
        if(h_C[i]!=3.0f){
            flag = false;
            break;
        }
    }
    cout << "flag="<<flag<<endl;
    return 0;
}