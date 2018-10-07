#include "w_update.cuh"


__host__
void cpu_w_update_kernel( float *w,
                          float *w_grad,
                          unsigned int size,

                          float *m,
                          float *v,

                          float learning_rate,
                          float beta1,
                          float beta2,
                          float epsilon)
{
  for (unsigned int w_ptr = 0; w_ptr <size; w_ptr++)
  {
    float w_dif = w_grad[w_ptr];

    #ifdef NETWORK_USE_ADAM_OPTIMIZER

      m[w_ptr] = beta1*m[w_ptr] + (1.0 - beta1)*w_dif;
      v[w_ptr] = beta2*v[w_ptr] + (1.0 - beta2)*w_dif*w_dif;

      float m_ = m[w_ptr]/(1.0 - beta1);
      float v_ = v[w_ptr]/(1.0 - beta2);

      w_dif = m_/(sqrt(v_) + epsilon);

    #endif

    w[w_ptr]+= learning_rate*w_dif;
  }
}


__global__
void cuda_w_update_kernel(  float *w,
                            float *w_grad,
                            unsigned int size,

                            float *m,
                            float *v,

                            float learning_rate,
                            float beta1,
                            float beta2,
                            float epsilon)
{
  unsigned int w_ptr   = threadIdx.x + blockIdx.x*blockDim.x;

  if (w_ptr < size)
  {
    float w_dif = w_grad[w_ptr];

    #ifdef NETWORK_USE_ADAM_OPTIMIZER

      m[w_ptr] = beta1*m[w_ptr] + (1.0 - beta1)*w_dif;
      v[w_ptr] = beta2*v[w_ptr] + (1.0 - beta2)*w_dif*w_dif;

      float m_ = m[w_ptr]/(1.0 - beta1);
      float v_ = v[w_ptr]/(1.0 - beta2);

      w_dif = m_/(sqrt(v_) + epsilon);

    #endif

    w[w_ptr]+= learning_rate*w_dif;
  }
}

void w_update(Tensor &w, Tensor &w_grad, Tensor &m, Tensor &v,
              sHyperparameters hyperparameters)
{
  unsigned int size = w.size();

  #ifdef NETWORK_USE_CUDA

    dim3 block(256);
    dim3 grid((size  + block.x - 1)/block.x);

    cuda_w_update_kernel<<<grid, block>>>(  w.v,
                                            w_grad.v,
                                            size,

                                            m.v,
                                            v.v,

                                            hyperparameters.learning_rate,
                                            hyperparameters.beta1,
                                            hyperparameters.beta2,
                                            hyperparameters.epsilon );

    cudaDeviceSynchronize();

  #else

    cpu_w_update_kernel(  w.v,
                          w_grad.v,
                          size,

                          m.v,
                          v.v,

                          hyperparameters.learning_rate,
                          hyperparameters.beta1,
                          hyperparameters.beta2,
                          hyperparameters.epsilon );

  #endif

  w.regularization_l1(hyperparameters.lambda1*hyperparameters.minibatch_size);
  w.regularization_l2(hyperparameters.lambda2*hyperparameters.minibatch_size);
}
