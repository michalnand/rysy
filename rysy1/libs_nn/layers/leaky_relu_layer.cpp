#include "leaky_relu_layer.h"
#include "kernels/leaky_relu_layer.cuh"
 
LeakyReluLayer::LeakyReluLayer()
        :Layer()
{

}

LeakyReluLayer::LeakyReluLayer(LeakyReluLayer& other)
        :Layer(other)
{
  copy_leaky_relu(other);
}

LeakyReluLayer::LeakyReluLayer(const LeakyReluLayer& other)
        :Layer(other)
{
  copy_leaky_relu(other);
}

LeakyReluLayer::~LeakyReluLayer()
{

}

LeakyReluLayer& LeakyReluLayer::operator= (LeakyReluLayer& other)
{
  copy(other);
  copy_leaky_relu(other);
  return *this;
}

LeakyReluLayer& LeakyReluLayer::operator= (const LeakyReluLayer& other)
{
  copy(other);
  copy_leaky_relu(other);
  return *this;
}

LeakyReluLayer::LeakyReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  layer_name = "LEAKY RELU";
}

void LeakyReluLayer::copy_leaky_relu(LeakyReluLayer &other)
{
  (void)other;
}

void LeakyReluLayer::copy_leaky_relu(const LeakyReluLayer &other)
{
  (void)other;
}


void LeakyReluLayer::forward(Tensor &output, Tensor &input)
{
  leaky_relu_layer_forward(output, input);
}

void LeakyReluLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  leaky_relu_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
