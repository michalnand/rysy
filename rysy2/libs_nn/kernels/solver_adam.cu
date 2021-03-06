#include "solver_adam.cuh"

__host__
void cpu_solver_adam_kernel( float *w,
                          float *w_grad,
                          unsigned int size,

                          float *m,
                          float *v,

                          float learning_rate,
                          float beta1,
                          float beta2,
                          float epsilon,
                          float gradient_clip)
{
    for (unsigned int w_ptr = 0; w_ptr <size; w_ptr++)
    {
        float w_dif = w_grad[w_ptr];

        if (w_dif > gradient_clip)
            w_dif = gradient_clip;
        if (w_dif < -gradient_clip)
            w_dif = -gradient_clip;

        m[w_ptr] = beta1*m[w_ptr] + (1.0 - beta1)*w_dif;
        v[w_ptr] = beta2*v[w_ptr] + (1.0 - beta2)*w_dif*w_dif;

        float m_ = m[w_ptr]/(1.0 - beta1);
        float v_ = v[w_ptr]/(1.0 - beta2);

        w_dif = m_/(sqrt(v_) + epsilon);


        w[w_ptr]+= learning_rate*w_dif;
    }
}


__global__
void cuda_solver_adam_kernel(   float *w,
                                float *w_grad,
                                unsigned int size,

                                float *m,
                                float *v,

                                float learning_rate,
                                float beta1,
                                float beta2,
                                float epsilon,
                                float gradient_clip)
{
    unsigned int w_ptr   = threadIdx.x + blockIdx.x*blockDim.x;

    if (w_ptr < size)
    {
        float w_dif = w_grad[w_ptr];

        if (w_dif > gradient_clip)
            w_dif = gradient_clip;
        if (w_dif < -gradient_clip)
            w_dif = -gradient_clip;

        m[w_ptr] = beta1*m[w_ptr] + (1.0 - beta1)*w_dif;
        v[w_ptr] = beta2*v[w_ptr] + (1.0 - beta2)*w_dif*w_dif;

        float m_ = m[w_ptr]/(1.0 - beta1);
        float v_ = v[w_ptr]/(1.0 - beta2);

        w_dif = m_/(sqrt(v_) + epsilon);


        w[w_ptr]+= learning_rate*w_dif;
    }
}



__host__
void cpu_regularization_kernel( float *w,
                                unsigned int size,
                                float lambda1,
                                float lambda2,
                                float gradient_clip )
{
    for (unsigned int w_ptr = 0; w_ptr <size; w_ptr++)
    {
        float value = w[w_ptr];

        if (value > 0.0)
            value-= lambda1;
        else
            value+= lambda1;

        value-= lambda2*value;

        w[w_ptr] = value;
    }
}


__global__
void cuda_regularization_kernel(    float *w,
                                    unsigned int size,
                                    float lambda1,
                                    float lambda2 )
{
    unsigned int w_ptr   = threadIdx.x + blockIdx.x*blockDim.x;

    if (w_ptr < size)
    {
        float value = w[w_ptr];

        if (value > 0.0)
            value-= lambda1;
        else
            value+= lambda1;

        value-= lambda2*value;

        w[w_ptr] = value;
    }
}

void solver_adam(   Tensor &w, Tensor &w_grad, Tensor &m, Tensor &v,
                    float learning_rate, float lambda1, float lambda2, float gradient_clip)
{
    unsigned int size = w.size();

    /*
    {
        std::vector<float> d_grad(size);
        w_grad.set_to_host(d_grad);

        float max = d_grad[0];
        float min = d_grad[0];
        for (unsigned int i = 0; i < d_grad.size(); i++)
        {
            if (d_grad[i] > max)
                max = d_grad[i];
            if (d_grad[i] < min)
                min = d_grad[i];
        }

        std::cout << "GRADIENT " << min << " " << max << "\n";
    }
    */
    
    float beta1     = 0.9;
    float beta2     = 0.999;
    float epsilon   = 0.00000001;

    #ifdef NETWORK_USE_CUDA
    {
        dim3 block(16);
        dim3 grid((size  + block.x + 1)/block.x);

        cuda_solver_adam_kernel<<<grid, block>>>(  w.v,
                                                w_grad.v,
                                                size,

                                                m.v,
                                                v.v,

                                                learning_rate,
                                                beta1,
                                                beta2,
                                                epsilon,
                                                gradient_clip);

        cudaDeviceSynchronize();
    }
    #else
    {
        cpu_solver_adam_kernel(  w.v,
                              w_grad.v,
                              size,

                              m.v,
                              v.v,

                              learning_rate,
                              beta1,
                              beta2,
                              epsilon,
                              gradient_clip);
    }

    #endif

    #ifdef NETWORK_USE_CUDA
    {
        dim3 block(16);
        dim3 grid((size  + block.x + 1)/block.x);

        cuda_regularization_kernel<<<grid, block>>>(    w.v,
                                                        size,
                                                        lambda1,
                                                        lambda2);

        cudaDeviceSynchronize();
    }
    #else
    {
        cpu_regularization_kernel(  w.v,
                                    size,
                                    lambda1,
                                    lambda2);
    }
    #endif
}
