#include "cuda_float_allocator.cuh"
#include <iostream>

/*
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
*/




std::mutex cu_allocator_mutex;

 

float* cu_malloc(unsigned int count)
{
  float *result = nullptr;

  cu_allocator_mutex.lock();

  cudaMalloc(&result, count*sizeof(float));

  cu_allocator_mutex.unlock();

  cu_clear(result, count);

  return result;
}

void cu_free(void *ptr)
{
    cu_allocator_mutex.lock();

  if (ptr != nullptr)
  {
    cudaFree(ptr);
    ptr = nullptr;
  }

  cu_allocator_mutex.unlock();
}

void cu_host_to_device(float *dev_ptr, float *host_ptr, unsigned int size)
{
  cudaMemcpy(dev_ptr, host_ptr, size*sizeof(float), cudaMemcpyHostToDevice);
}

void cu_device_to_host(float *host_ptr, float *dev_ptr, unsigned int size)
{
  cudaMemcpy(host_ptr, dev_ptr, size*sizeof(float), cudaMemcpyDeviceToHost);
}

void cu_device_to_device(float *dest_ptr, float *src_ptr, unsigned int size)
{
  cudaMemcpy(dest_ptr, src_ptr, size*sizeof(float), cudaMemcpyDeviceToDevice);
}

void cu_clear(float *result, unsigned int size)
{
  cudaMemset(result, 0, size*sizeof(float));
}

size_t cu_get_mem_free()
{
    cu_allocator_mutex.lock();

    size_t mem_free = 0;
    size_t mem_total = 0;
    cudaMemGetInfo(&mem_free, &mem_total);

    cu_allocator_mutex.unlock();

    return mem_free;
}
