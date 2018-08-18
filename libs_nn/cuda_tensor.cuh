#ifndef _CUDA_TENSOR_CUH_
#define _CUDA_TENSOR_CUH_

void cuda_tensor_clear(float *v, unsigned int size);
void cuda_tensor_set_const(float *v, unsigned int size, float value);
void cuda_tensor_random(float *v, unsigned int size, float range);


void cuda_tensor_add(float *result, float *rhs, unsigned int size);
void cuda_tensor_sub(float *result, float *rhs, unsigned int size);
void cuda_tensor_mul(float *result, float value, unsigned int size);


void cuda_tensor_set_element(float *result, float value, unsigned int idx);
float cuda_tensor_get_element(float *src_ptr, unsigned int idx);

void cuda_tensor_regularization_l1(float *result, float lambda, unsigned int size);
void cuda_tensor_regularization_l2(float *result, float lambda, unsigned int size);


void cuda_rms(float *result, float *va, float *vb, unsigned int size);

#endif
