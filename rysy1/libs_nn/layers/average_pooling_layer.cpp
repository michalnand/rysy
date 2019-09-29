#include "average_pooling_layer.h"
#include "kernels/average_pooling_layer.cuh"

AveragePoolingLayer::AveragePoolingLayer()
        :Layer()
{

}

AveragePoolingLayer::AveragePoolingLayer(AveragePoolingLayer& other)
        :Layer(other)
{
  copy_max_pooling(other);
}

AveragePoolingLayer::AveragePoolingLayer(const AveragePoolingLayer& other)
        :Layer(other)
{
  copy_max_pooling(other);
}

AveragePoolingLayer::~AveragePoolingLayer()
{

}

AveragePoolingLayer& AveragePoolingLayer::operator= (AveragePoolingLayer& other)
{
  copy(other);
  copy_max_pooling(other);
  return *this;
}

AveragePoolingLayer& AveragePoolingLayer::operator= (const AveragePoolingLayer& other)
{
  copy(other);
  copy_max_pooling(other);
  return *this;
}

AveragePoolingLayer::AveragePoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = kernel_geometry.w;
  this->kernel_geometry.h = kernel_geometry.h;
  this->kernel_geometry.d = 1;

  this->output_geometry.w = this->input_geometry.w/this->kernel_geometry.w;
  this->output_geometry.h = this->input_geometry.h/this->kernel_geometry.h;
  this->output_geometry.d = this->input_geometry.d;

  unsigned int output_size = this->output_geometry.w*this->output_geometry.h*this->output_geometry.d;
               output_size*= this->kernel_geometry.w*this->kernel_geometry.h;

  flops = output_size*2;

  layer_name = "AVERAGE POOLING";
}

void AveragePoolingLayer::copy_max_pooling(AveragePoolingLayer &other)
{
    (void)other;
}

void AveragePoolingLayer::copy_max_pooling(const AveragePoolingLayer &other)
{
    (void)other;
}


void AveragePoolingLayer::forward(Tensor &output, Tensor &input)
{
    average_pooling_layer_forward(output, input);
}

void AveragePoolingLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
    (void)update_weights;
    average_pooling_layer_backward(layer_mem_prev.error, layer_mem.error);
}
