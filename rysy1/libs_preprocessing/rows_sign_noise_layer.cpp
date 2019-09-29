#include "rows_sign_noise_layer.h"
#include "kernels/rows_sign_noise_layer.cuh"


RowsSignNoiseLayer::RowsSignNoiseLayer()
              :PreprocessingLayer()
{

}

RowsSignNoiseLayer::RowsSignNoiseLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{

}

RowsSignNoiseLayer::~RowsSignNoiseLayer()
{

}


void RowsSignNoiseLayer::process(Tensor &output, Tensor &input, unsigned int augumentation)
{
  (void)augumentation;
  
  if (output.h() != noise.h())
  {
    noise.init(output.h());
  }

  noise.set_random(1.0);

  rows_sign_noise_layer(output, input, noise);
}
