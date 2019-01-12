#include "relu6_layer.h"
#include "kernels/relu6_layer.cuh"

Relu6Layer::Relu6Layer()
        :Layer()
{

}

Relu6Layer::Relu6Layer(Relu6Layer& other)
        :Layer(other)
{
  copy_relu6(other);
}

Relu6Layer::Relu6Layer(const Relu6Layer& other)
        :Layer(other)
{
  copy_relu6(other);
}

Relu6Layer::~Relu6Layer()
{

}

Relu6Layer& Relu6Layer::operator= (Relu6Layer& other)
{
  copy(other);
  copy_relu6(other);
  return *this;
}

Relu6Layer& Relu6Layer::operator= (const Relu6Layer& other)
{
  copy(other);
  copy_relu6(other);
  return *this;
}

Relu6Layer::Relu6Layer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  layer_name = "RELU6";
}

void Relu6Layer::copy_relu6(Relu6Layer &other)
{
  (void)other;
}

void Relu6Layer::copy_relu6(const Relu6Layer &other)
{
  (void)other;
}


void Relu6Layer::forward(Tensor &output, Tensor &input)
{
  relu_6_layer_forward(output, input);
}

void Relu6Layer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  relu_6_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
