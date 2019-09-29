#include "relu_layer.h"
#include "kernels/relu_layer.cuh"

ReluLayer::ReluLayer()
        :Layer()
{

}

ReluLayer::ReluLayer(ReluLayer& other)
        :Layer(other)
{
  copy_relu(other);
}

ReluLayer::ReluLayer(const ReluLayer& other)
        :Layer(other)
{
  copy_relu(other);
}

ReluLayer::~ReluLayer()
{

}

ReluLayer& ReluLayer::operator= (ReluLayer& other)
{
  copy(other);
  copy_relu(other);
  return *this;
}

ReluLayer& ReluLayer::operator= (const ReluLayer& other)
{
  copy(other);
  copy_relu(other);
  return *this;
}

ReluLayer::ReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  layer_name = "RELU";
}

void ReluLayer::copy_relu(ReluLayer &other)
{
  (void)other;
}

void ReluLayer::copy_relu(const ReluLayer &other)
{
  (void)other;
}


void ReluLayer::forward(Tensor &output, Tensor &input)
{
  relu_layer_forward(output, input);
}

void ReluLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  relu_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
