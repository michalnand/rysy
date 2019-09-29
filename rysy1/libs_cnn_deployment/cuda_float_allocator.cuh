#ifndef _CUDA_FLOAT_ALLOCATOR_CUH_
#define _CUDA_FLOAT_ALLOCATOR_CUH_

#include <mutex>
#include <string>

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

      void load_from_file(float *ptr, std::string file_name, unsigned int size);

};


extern CudaFloatAllocator cuda_float_allocator;



#endif
