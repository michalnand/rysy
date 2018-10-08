#include "luma_noise_layer.h"
#include "kernels/luma_noise_layer.cuh"

#include <iostream>

LumaNoiseLayer::LumaNoiseLayer()
              :PreprocessingLayer()
{

}

LumaNoiseLayer::LumaNoiseLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{
  noise_level = parameters["noise"].asFloat();
}

LumaNoiseLayer::~LumaNoiseLayer()
{

}



void LumaNoiseLayer::process(Tensor &output, Tensor &input)
{
  float noise_f = (rand()%100000)/100000.0;
  if (rand()%2)
    noise_f = -noise_f;

  noise_f = noise_f*noise_level;

  luma_noise_layer(output, input, noise_f);
}
