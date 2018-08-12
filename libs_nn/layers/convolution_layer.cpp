#include "convolution_layer.h"

#include "kernels/convolution_layer.cuh"
#include "kernels/w_update.cuh"


ConvolutionLayer::ConvolutionLayer()
        :Layer()
{

}

ConvolutionLayer::ConvolutionLayer(ConvolutionLayer& other)
        :Layer(other)
{
  copy_convolution(other);
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& other)
        :Layer(other)
{
  copy_convolution(other);
}

ConvolutionLayer::~ConvolutionLayer()
{

}

ConvolutionLayer& ConvolutionLayer::operator= (ConvolutionLayer& other)
{
  copy(other);
  copy_convolution(other);
  return *this;
}

ConvolutionLayer& ConvolutionLayer::operator= (const ConvolutionLayer& other)
{
  copy(other);
  copy_convolution(other);
  return *this;
}

ConvolutionLayer::ConvolutionLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry.w = input_geometry.w;
  this->input_geometry.h = input_geometry.h;
  this->input_geometry.d = input_geometry.d;

  this->output_geometry.w = this->input_geometry.w;
  this->output_geometry.h = this->input_geometry.h;
  this->output_geometry.d = this->kernel_geometry.d;

  sGeometry tmp_kernel_geometry = kernel_geometry;
  tmp_kernel_geometry.d = this->input_geometry.d*this->output_geometry.d;

  w.init(tmp_kernel_geometry);

  w.set_random_xavier();

  w_grad.init(tmp_kernel_geometry);


  bias.init(this->output_geometry.d);
  bias.set_random(0.0000001);

  m.init(w.get_geometry());
  v.init(w.get_geometry());

  unsigned int output_size  = output_geometry.w*output_geometry.h;
  flops = output_size*kernel_geometry.d*(1 + kernel_geometry.w*kernel_geometry.h*input_geometry.d);

  layer_name = "CONVOLUTION";
}

void ConvolutionLayer::copy_convolution(ConvolutionLayer &other)
{
  (void)other;
}

void ConvolutionLayer::copy_convolution(const ConvolutionLayer &other)
{
  (void)other;
}


void ConvolutionLayer::forward(Tensor &output, Tensor &input)
{
  convolution_layer_forward(output, input, w, bias);
}


void ConvolutionLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  convolution_layer_gradient(w_grad, layer_mem_prev.output, layer_mem.error);

  convolution_layer_update_bias(bias, layer_mem.error, hyperparameters.learning_rate);

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }


  //backpropagate error
  convolution_layer_backward(layer_mem_prev.error, layer_mem_prev.output, layer_mem.error, w);
}
