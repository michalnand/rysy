#include "gating_layer.h"
#include "kernels/gating_layer.cuh"

GatingLayer::GatingLayer()
        :Layer()
{

}

GatingLayer::GatingLayer(GatingLayer& other)
        :Layer(other)
{
  copy_gating(other);
}

GatingLayer::GatingLayer(const GatingLayer& other)
        :Layer(other)
{
  copy_gating(other);
}

GatingLayer::~GatingLayer()
{

}

GatingLayer& GatingLayer::operator= (GatingLayer& other)
{
  copy(other);
  copy_gating(other);
  return *this;
}

GatingLayer& GatingLayer::operator= (const GatingLayer& other)
{
  copy(other);
  copy_gating(other);
  return *this;
}

GatingLayer::GatingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry.w = this->input_geometry.w;
  this->output_geometry.h = this->input_geometry.h;
  this->output_geometry.d = this->input_geometry.d/2;

  unsigned int input_size = input_geometry.w*input_geometry.h*input_geometry.d;
  flops = input_size;

  layer_name = "GATING";
}
 
void GatingLayer::copy_gating(GatingLayer &other)
{
  (void)other;
}

void GatingLayer::copy_gating(const GatingLayer &other)
{
  (void)other;
}


void GatingLayer::forward(Tensor &output, Tensor &input)
{
  gating_layer_forward(output, input);
}

void GatingLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  gating_layer_backward(layer_mem_prev.error, layer_mem_prev.output, layer_mem.output, layer_mem.error);
}
