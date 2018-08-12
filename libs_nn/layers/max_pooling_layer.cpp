#include "max_pooling_layer.h"
#include "kernels/max_pooling_layer.cuh"

MaxPoolingLayer::MaxPoolingLayer()
        :Layer()
{

}

MaxPoolingLayer::MaxPoolingLayer(MaxPoolingLayer& other)
        :Layer(other)
{
  copy_max_pooling(other);
}

MaxPoolingLayer::MaxPoolingLayer(const MaxPoolingLayer& other)
        :Layer(other)
{
  copy_max_pooling(other);
}

MaxPoolingLayer::~MaxPoolingLayer()
{

}

MaxPoolingLayer& MaxPoolingLayer::operator= (MaxPoolingLayer& other)
{
  copy(other);
  copy_max_pooling(other);
  return *this;
}

MaxPoolingLayer& MaxPoolingLayer::operator= (const MaxPoolingLayer& other)
{
  copy(other);
  copy_max_pooling(other);
  return *this;
}

MaxPoolingLayer::MaxPoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
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

  max_mask.init(this->input_geometry);

  layer_name = "MAX POOLING";
}

void MaxPoolingLayer::copy_max_pooling(MaxPoolingLayer &other)
{
  max_mask = other.max_mask;
}

void MaxPoolingLayer::copy_max_pooling(const MaxPoolingLayer &other)
{
  max_mask = other.max_mask;
}


void MaxPoolingLayer::forward(Tensor &output, Tensor &input)
{
  max_pooling_layer_forward(max_mask, output, input);
}

void MaxPoolingLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  max_pooling_layer_backward(layer_mem_prev.error, layer_mem.error, max_mask);
}
