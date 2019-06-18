#include "cuda_float_allocator.cuh"

#include <iostream>
#include <fstream>
#include <vector>


CudaFloatAllocator cuda_float_allocator;

CudaFloatAllocator::CudaFloatAllocator()
{

}

CudaFloatAllocator::~CudaFloatAllocator()
{

}

float* CudaFloatAllocator::malloc(unsigned int count)
{
    mutex.lock();
    float *result = nullptr;

    cudaMalloc(&result, count*sizeof(float));

    mutex.unlock();

    clear(result, count);

    return result;
}

void CudaFloatAllocator::free(void *ptr)
{
    mutex.lock();

    if (ptr != nullptr)
    {
        cudaFree(ptr);
        ptr = nullptr;
    }

    mutex.unlock();
}

void CudaFloatAllocator::host_to_device(float *dev_ptr, float *host_ptr, unsigned int size)
{
    cudaMemcpy(dev_ptr, host_ptr, size*sizeof(float), cudaMemcpyHostToDevice);
}

void CudaFloatAllocator::device_to_host(float *host_ptr, float *dev_ptr, unsigned int size)
{
    cudaMemcpy(host_ptr, dev_ptr, size*sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaFloatAllocator::device_to_device(float *dest_ptr, float *src_ptr, unsigned int size)
{
    cudaMemcpy(dest_ptr, src_ptr, size*sizeof(float), cudaMemcpyDeviceToDevice);
}

void CudaFloatAllocator::clear(float *result, unsigned int size)
{
    cudaMemset(result, 0, size*sizeof(float));
}



void CudaFloatAllocator::load_from_file(float *ptr, std::string file_name, unsigned int size)
{
    std::vector<float> tmp(size);
    std::ifstream input_file;

    input_file.open(file_name, std::ios::in | std::ios::binary);
    if (input_file.is_open())
    {
        std::cout << "loading file " << file_name << "\n";
        input_file.read( (char*)(&tmp[0]), sizeof(float)*size);
        input_file.close();

        host_to_device(ptr, &tmp[0], size);
    }
}
