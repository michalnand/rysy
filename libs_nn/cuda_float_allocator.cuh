#ifndef _CUDA_FLOAT_ALLOCATOR_CUH_
#define _CUDA_FLOAT_ALLOCATOR_CUH_

#include <mutex>

/*
class CudaFloatAllocator
{
  private:
    std::mutex mutex;

  public:
      CudaFloatAllocator();
      virtual ~CudaFloatAllocator();

      float* malloc(unsigned int count);
      void free(void *ptr);


  public:

      void host_to_device(float *dev_ptr, float *host_ptr, unsigned int size);
      void device_to_host(float *host_ptr, float *dev_ptr, unsigned int size);
      void device_to_device(float *dest_ptr, float *src_ptr, unsigned int size);

      void clear(float *result, unsigned int size);
};


extern CudaFloatAllocator cuda_float_allocator;
*/


float* cu_malloc(unsigned int count);
void cu_free(void *ptr);
void cu_host_to_device(float *dev_ptr, float *host_ptr, unsigned int size);
void cu_device_to_host(float *host_ptr, float *dev_ptr, unsigned int size);
void cu_device_to_device(float *dest_ptr, float *src_ptr, unsigned int size);

void cu_clear(float *result, unsigned int size);


#endif
