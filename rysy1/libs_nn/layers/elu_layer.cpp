#include "elu_layer.h"
#include "kernels/elu_layer.cuh"

EluLayer::EluLayer()
        :Layer()
{

}

EluLayer::EluLayer(EluLayer& other)
        :Layer(other)
{
  copy_elu(other);
}
 
EluLayer::EluLayer(const EluLayer& other)
        :Layer(other)
{
  copy_elu(other);
}

EluLayer::~EluLayer()
{

}

EluLayer& EluLayer::operator= (EluLayer& other)
{
  copy(other);
  copy_elu(other);
  return *this;
}

EluLayer& EluLayer::operator= (const EluLayer& other)
{
  copy(other);
  copy_elu(other);
  return *this;
}

EluLayer::EluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  layer_name = "ELU";
}

void EluLayer::copy_elu(EluLayer &other)
{
  (void)other;
}

void EluLayer::copy_elu(const EluLayer &other)
{
  (void)other;
}


void EluLayer::forward(Tensor &output, Tensor &input)
{
  elu_layer_forward(output, input);
}

void EluLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  elu_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
