#include "dropout_layer.h"
#include "kernels/dropout_layer.cuh"

DropoutLayer::DropoutLayer()
        :Layer()
{

}

DropoutLayer::DropoutLayer(DropoutLayer& other)
        :Layer(other)
{
  copy_dropout(other);
}

DropoutLayer::DropoutLayer(const DropoutLayer& other)
        :Layer(other)
{
  copy_dropout(other);
}

DropoutLayer::~DropoutLayer()
{

}

DropoutLayer& DropoutLayer::operator= (DropoutLayer& other)
{
  copy(other);
  copy_dropout(other);
  return *this;
}

DropoutLayer& DropoutLayer::operator= (const DropoutLayer& other)
{
  copy(other);
  copy_dropout(other);
  return *this;
}

DropoutLayer::DropoutLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  noise.init(output_geometry);

  layer_name = "DROPOUT";
}

void DropoutLayer::copy_dropout(DropoutLayer &other)
{
  (void)other;
}

void DropoutLayer::copy_dropout(const DropoutLayer &other)
{
  (void)other;
}


void DropoutLayer::forward(Tensor &output, Tensor &input)
{
    if (is_training_mode())
    {
        noise.set_random(1.0);
        dropout_layer_forward(output, input, noise, hyperparameters.dropout);
    }
    else
    {
        output.copy(input);
        output.mul(1.0 - hyperparameters.dropout);
    }
}

void DropoutLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  dropout_layer_backward(layer_mem_prev.error, layer_mem.output, layer_mem.error);
}
