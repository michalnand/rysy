#include "fc_layer.cuh"

__host__
void cpu_fc_forward_kernel(   float *output,
                              float *input,
                              float *w,
                              float *bias,
                              unsigned int output_size,
                              unsigned int input_size )
{
  for (unsigned int neuron_idx = 0; neuron_idx < output_size; neuron_idx++)
  {
    unsigned int w_ptr = neuron_idx*input_size;

    float sum = bias[neuron_idx];
    for (unsigned int i = 0; i < input_size; i++)
    {
      sum+= w[w_ptr]*input[i];
      w_ptr++;
    }

    output[neuron_idx] = sum;
  }
}


__global__
void cuda_fc_forward_kernel(  float *output,
                              float *input,
                              float *w,

                              unsigned int neuron_synaptic_paralelisation,
                              unsigned int neuron_synaptic_paralelisation_size,
                              unsigned int input_size,
                              unsigned int neurons_count)
{
    unsigned int synapse_part_idx  = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int neuron_idx        = threadIdx.y + blockIdx.y*blockDim.y;

    if (neuron_idx < neurons_count)
    if (synapse_part_idx < neuron_synaptic_paralelisation)
    {
      w     = w     + neuron_synaptic_paralelisation_size*synapse_part_idx  + neuron_idx*input_size;
      input = input + neuron_synaptic_paralelisation_size*synapse_part_idx;

      float sum = 0.0;

      for (unsigned int i = 0; i < neuron_synaptic_paralelisation_size; i++)
        sum+= w[i]*input[i];

      atomicAdd(&output[neuron_idx], sum);
    }
}



void fc_layer_forward(  Tensor &output, Tensor &input,
                        Tensor &w, Tensor &bias)
{
  unsigned int input_size   = input.size();
  unsigned int output_size  = output.size();

  #ifdef NETWORK_USE_CUDA
      unsigned int neuron_synaptic_paralelisation = sqrt(input_size);

      while ((input_size%neuron_synaptic_paralelisation) != 0)
        neuron_synaptic_paralelisation--;

      unsigned int neuron_synaptic_paralelisation_size  = input_size/neuron_synaptic_paralelisation;

      cudaMemcpy(output.v, bias.v, output_size*sizeof(float), cudaMemcpyDeviceToDevice);

      dim3 block(8, 8);
      dim3 grid((neuron_synaptic_paralelisation_size  + block.x - 1)/block.x,
                (output_size  + block.y - 1)/block.y);


        cuda_fc_forward_kernel<<<grid, block>>>(  output.v,
                                                  input.v,
                                                  w.v,

                                                  neuron_synaptic_paralelisation,
                                                  neuron_synaptic_paralelisation_size,
                                                  input_size,
                                                  output_size
                                                 );
        cudaDeviceSynchronize();

  #else

       cpu_fc_forward_kernel( output.v,
                              input.v,
                              w.v,
                              bias.v,
                              output_size,
                              input_size );

  #endif
}








__host__
void cpu_fc_layer_back_kernel(  float *error,
                                 float *error_back,
                                 float *w,
                                 unsigned int input_size,
                                 unsigned int output_size)
{
  for (unsigned int input_idx = 0; input_idx < input_size; input_idx++)
  {
    unsigned int w_ptr = input_idx;

    float err = 0.0;

    for (unsigned int i = 0; i < output_size; i++)
    {
      err+= error[i]*w[w_ptr];
      w_ptr+= input_size;
    }

    error_back[input_idx] = err;
  }
}

__global__
void cuda_fc_layer_back_kernel(  float *error,
                                 float *error_back,
                                 float *w,
                                 unsigned int input_size,
                                 unsigned int output_size)
{
  unsigned int input_idx   = threadIdx.x + blockIdx.x*blockDim.x;

  if (input_idx < input_size)
  {
    unsigned int w_ptr = input_idx;

    float err = 0.0;

    for (unsigned int i = 0; i < output_size; i++)
    {
      err+= error[i]*w[w_ptr];
      w_ptr+= input_size;
    }

    error_back[input_idx] = err;
  }
}

//error backpropagation
void fc_layer_backward( Tensor &error_back, Tensor &input, Tensor &error, Tensor &w)
{
  unsigned int input_size   = input.size();
  unsigned int output_size  = error.size();


  #ifdef NETWORK_USE_CUDA

      {
        cudaMemset(error_back.v, 0, error_back.size()*sizeof(float));

        dim3 block(16);
        dim3 grid( (input_size  + block.x - 1)/block.x);

        cuda_fc_layer_back_kernel<<<grid, block>>>( error.v,
                                                    error_back.v,
                                                    w.v,
                                                    input_size,
                                                    output_size);
        cudaDeviceSynchronize();
      }

  #else

   cpu_fc_layer_back_kernel(  error.v,
                              error_back.v,
                              w.v,
                              input_size,
                              output_size);

  #endif
}







//learn weights

__host__
void cpu_fc_layer_weights_gradient( float *w_grad,
                                    float *error,
                                    float *input,

                                    unsigned int input_size,
                                    unsigned int neurons_count )
{
  for (unsigned int  neuron_idx = 0; neuron_idx < neurons_count; neuron_idx++)
  for (unsigned int  input_idx  = 0; input_idx < input_size; input_idx++)
  {
    unsigned int w_ptr = neuron_idx*input_size + input_idx;
    w_grad[w_ptr]+= error[neuron_idx]*input[input_idx];
  }
}

__global__
void cuda_fc_layer_weights_gradient(  float *w_grad,
                                      float *error,
                                      float *input,

                                      unsigned int input_size,
                                      unsigned int neurons_count )
{
  unsigned int input_idx   = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int neuron_idx  = threadIdx.y + blockIdx.y*blockDim.y;

  if (neuron_idx < neurons_count)
  if (input_idx < input_size)
  {
    unsigned int w_ptr = neuron_idx*input_size + input_idx;
    w_grad[w_ptr]+= error[neuron_idx]*input[input_idx];
  }
}

void fc_layer_gradient(Tensor &w_grad, Tensor &input, Tensor &error)
{
  unsigned int input_size   = input.size();
  unsigned int output_size  = error.size();

  #ifdef NETWORK_USE_CUDA

      dim3 block(16, 16);
      dim3 grid( (input_size  + block.x - 1)/block.x,
                 (output_size  + block.y - 1)/block.y);

      cuda_fc_layer_weights_gradient<<<grid, block>>>(  w_grad.v,
                                                        error.v,
                                                        input.v,

                                                        input_size,
                                                        output_size );
      cudaDeviceSynchronize();

  #else

      cpu_fc_layer_weights_gradient(  w_grad.v,
                                      error.v,
                                      input.v,

                                      input_size,
                                      output_size );

  #endif

}




__host__
void cpu_fc_layer_back_bias_kernel(   float *error,
                                       float *bias,
                                       float learning_rate,
                                       unsigned int output_size)
{
  for (unsigned int neuron_idx = 0; neuron_idx < output_size; neuron_idx++)
  {
    float b_dif = error[neuron_idx]*learning_rate;
    bias[neuron_idx]+= b_dif;
  }
}

__global__
void cuda_fc_layer_back_bias_kernel(   float *error,
                                       float *bias,
                                       float learning_rate,
                                       unsigned int output_size)
{
  unsigned int neuron_idx  = threadIdx.x + blockIdx.x*blockDim.x;

  if (neuron_idx < output_size)
  {
    float b_dif = error[neuron_idx]*learning_rate;
    bias[neuron_idx]+= b_dif;
  }
}

void fc_layer_update_bias(Tensor &bias, Tensor &error, float learning_rate)
{
  #ifdef NETWORK_USE_CUDA

    dim3 block(16);
    dim3 grid((error.size()  + block.x - 1)/block.x);

    cuda_fc_layer_back_bias_kernel<<<grid, block>>>(  error.v,
                                                      bias.v,
                                                      learning_rate,
                                                      error.size());

    cudaDeviceSynchronize();
  #else

    cpu_fc_layer_back_bias_kernel( error.v,
                                   bias.v,
                                   learning_rate,
                                   error.size());

  #endif
}
