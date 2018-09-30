#include "noise_layer.h"
#include "kernels/noise_layer.cuh"

NoiseLayer::NoiseLayer()
        :Layer()
{

}

NoiseLayer::NoiseLayer(NoiseLayer& other)
        :Layer(other)
{
  copy_noise(other);
}

NoiseLayer::NoiseLayer(const NoiseLayer& other)
        :Layer(other)
{
  copy_noise(other);
}

NoiseLayer::~NoiseLayer()
{

}

NoiseLayer& NoiseLayer::operator= (NoiseLayer& other)
{
  copy(other);
  copy_noise(other);
  return *this;
}

NoiseLayer& NoiseLayer::operator= (const NoiseLayer& other)
{
  copy(other);
  copy_noise(other);
  return *this;
}

NoiseLayer::NoiseLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry = input_geometry;

  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = 1;

  this->output_geometry = this->input_geometry;

  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*2;

  white_noise.init(input_geometry);
  salt_and_pepper_noise.init(input_geometry);

  layer_name = "NOISE";
}

void NoiseLayer::copy_noise(NoiseLayer &other)
{
  (void)other;
}

void NoiseLayer::copy_noise(const NoiseLayer &other)
{
  (void)other;
}


void NoiseLayer::forward(Tensor &output, Tensor &input)
{
  white_noise.set_random(1.0);
  salt_and_pepper_noise.set_random(1.0);
  float brightness_noise = ( ((rand()%10000)/10000.0) - 0.5)*2.0;


  noise_layer_forward(output, input, hyperparameters, white_noise, salt_and_pepper_noise, brightness_noise);
}

void NoiseLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  layer_mem_prev.error.copy(layer_mem.error);
}
