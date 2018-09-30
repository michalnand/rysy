#include "dense_convolution_layer.h"

#include "kernels/convolution_layer_forward.cuh"
#include "kernels/convolution_layer_backward.cuh"
#include "kernels/w_update.cuh"

#include "../cuda_float_allocator.cuh"
#include "../cuda_tensor.cuh"

DenseConvolutionLayer::DenseConvolutionLayer()
        :Layer()
{

}

DenseConvolutionLayer::DenseConvolutionLayer(DenseConvolutionLayer& other)
        :Layer(other)
{
  copy_dense_convolution(other);
}

DenseConvolutionLayer::DenseConvolutionLayer(const DenseConvolutionLayer& other)
        :Layer(other)
{
  copy_dense_convolution(other);
}

DenseConvolutionLayer::~DenseConvolutionLayer()
{

}

DenseConvolutionLayer& DenseConvolutionLayer::operator= (DenseConvolutionLayer& other)
{
  copy(other);
  copy_dense_convolution(other);
  return *this;
}

DenseConvolutionLayer& DenseConvolutionLayer::operator= (const DenseConvolutionLayer& other)
{
  copy(other);
  copy_dense_convolution(other);
  return *this;
}

DenseConvolutionLayer::DenseConvolutionLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->output_geometry.w = this->input_geometry.w;
  this->output_geometry.h = this->input_geometry.h;
  this->output_geometry.d = this->kernel_geometry.d + this->input_geometry.d;

  sGeometry convolution_output_geometry;
  convolution_output_geometry.w = this->input_geometry.w;
  convolution_output_geometry.h = this->input_geometry.h;
  convolution_output_geometry.d = this->kernel_geometry.d;

  sGeometry tmp_kernel_geometry = kernel_geometry;
  tmp_kernel_geometry.d = this->input_geometry.d*convolution_output_geometry.d;

  w.init(tmp_kernel_geometry);
  w.set_random_xavier();

  w_grad.init(tmp_kernel_geometry);

  bias.init(convolution_output_geometry.d);
  bias.set_random(0.0000001);

  m.init(w.get_geometry());
  v.init(w.get_geometry());

  error_convolution.init(convolution_output_geometry);
  output_convolution.init(convolution_output_geometry);

  unsigned int output_size  = output_geometry.w*output_geometry.h;
  flops = output_size*kernel_geometry.d*(1 + kernel_geometry.w*kernel_geometry.h*input_geometry.d);

  layer_name = "DENSE CONVOLUTION";
}

void DenseConvolutionLayer::copy_dense_convolution(DenseConvolutionLayer &other)
{
  error_convolution   = other.error_convolution;
  output_convolution  = other.output_convolution;
}

void DenseConvolutionLayer::copy_dense_convolution(const DenseConvolutionLayer &other)
{
  error_convolution   = other.error_convolution;
  output_convolution  = other.output_convolution;
}


void DenseConvolutionLayer::forward(Tensor &output, Tensor &input)
{
  convolution_layer_forward(output_convolution, input, w, bias);

  cuda_float_allocator.device_to_device(output.v, output_convolution.v, output_convolution.size());
  cuda_float_allocator.device_to_device(output.v + output_convolution.size(), input.v, input.size());

}

void DenseConvolutionLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  cuda_float_allocator.device_to_device(error_convolution.v, layer_mem.error.v, error_convolution.size());

  //MINIBATCH
  convolution_layer_gradient(w_grad, layer_mem_prev.output, error_convolution);

  convolution_layer_update_bias(bias, error_convolution, hyperparameters.learning_rate);

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }


  //backpropagate error
  convolution_layer_backward( layer_mem_prev.error, layer_mem_prev.output, error_convolution, w);
  cuda_tensor_add(layer_mem_prev.error.v, layer_mem.error.v + error_convolution.size(), layer_mem_prev.error.size());

}
