#include "unpooling_layer.h"
#include "kernels/unpooling_layer.cuh"

UnPoolingLayer::UnPoolingLayer()
        :Layer()
{

}

UnPoolingLayer::UnPoolingLayer(UnPoolingLayer& other)
        :Layer(other)
{
  copy_unpooling(other);
}

UnPoolingLayer::UnPoolingLayer(const UnPoolingLayer& other)
        :Layer(other)
{
  copy_unpooling(other);
}

UnPoolingLayer::~UnPoolingLayer()
{


}

UnPoolingLayer& UnPoolingLayer::operator= (UnPoolingLayer& other)
{
  copy(other);
  copy_unpooling(other);
  return *this;
}

UnPoolingLayer& UnPoolingLayer::operator= (const UnPoolingLayer& other)
{
  copy(other);
  copy_unpooling(other);
  return *this;
}

UnPoolingLayer::UnPoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = kernel_geometry.w;
  this->kernel_geometry.h = kernel_geometry.h;
  this->kernel_geometry.d = 1;

  this->output_geometry.w = this->input_geometry.w*this->kernel_geometry.w;
  this->output_geometry.h = this->input_geometry.h*this->kernel_geometry.h;
  this->output_geometry.d = this->input_geometry.d;

  unsigned int output_size = this->input_geometry.w*this->input_geometry.h*this->input_geometry.d;
               output_size*= this->kernel_geometry.w*this->kernel_geometry.h;

  flops = output_size*2;

  layer_name = "UN POOLING";
}

void UnPoolingLayer::copy_unpooling(UnPoolingLayer &other)
{
  (void)other;
}

void UnPoolingLayer::copy_unpooling(const UnPoolingLayer &other)
{
  (void)other;
}


void UnPoolingLayer::forward(Tensor &output, Tensor &input)
{
  unpooling_layer_forward(output, input);
}

void UnPoolingLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  unpooling_layer_backward(layer_mem_prev.error, layer_mem.error);
}
