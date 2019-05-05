#include "saturated_leaky_relu_layer.h"
#include "kernels/saturated_leaky_relu_layer.cuh"

SaturatedLeakyReluLayer::SaturatedLeakyReluLayer()
        :Layer()
{

}

SaturatedLeakyReluLayer::SaturatedLeakyReluLayer(SaturatedLeakyReluLayer& other)
        :Layer(other)
{
  copy_saturated_leaky_relu(other);
}

SaturatedLeakyReluLayer::SaturatedLeakyReluLayer(const SaturatedLeakyReluLayer& other)
        :Layer(other)
{
  copy_saturated_leaky_relu(other);
}

SaturatedLeakyReluLayer::~SaturatedLeakyReluLayer()
{

}

SaturatedLeakyReluLayer& SaturatedLeakyReluLayer::operator= (SaturatedLeakyReluLayer& other)
{
  copy(other);
  copy_saturated_leaky_relu(other);
  return *this;
}

SaturatedLeakyReluLayer& SaturatedLeakyReluLayer::operator= (const SaturatedLeakyReluLayer& other)
{
  copy(other);
  copy_saturated_leaky_relu(other);
  return *this;
}

SaturatedLeakyReluLayer::SaturatedLeakyReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  layer_name = "SLRELU";
}

void SaturatedLeakyReluLayer::copy_saturated_leaky_relu(SaturatedLeakyReluLayer &other)
{
  (void)other;
}

void SaturatedLeakyReluLayer::copy_saturated_leaky_relu(const SaturatedLeakyReluLayer &other)
{
  (void)other;
}


void SaturatedLeakyReluLayer::forward(Tensor &output, Tensor &input)
{
  saturated_leaky_relu_layer_forward(output, input);
}

void SaturatedLeakyReluLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  saturated_leaky_relu_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
