#include "white_noise_layer.h"
#include "kernels/white_noise_layer.cuh"


WhiteNoiseLayer::WhiteNoiseLayer()
              :PreprocessingLayer()
{

}

WhiteNoiseLayer::WhiteNoiseLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{
  noise_level = parameters["noise"].asFloat();
}

WhiteNoiseLayer::~WhiteNoiseLayer()
{

}


void WhiteNoiseLayer::process(Tensor &output, Tensor &input)
{
  if ( (output.w() != noise.w())||(output.h() != noise.h())||(output.d() != noise.d()) )
  {
    noise.init(output.get_geometry());
  }


  noise.set_random(1.0); 

  white_noise_layer(output, input, noise, noise_level);
}
