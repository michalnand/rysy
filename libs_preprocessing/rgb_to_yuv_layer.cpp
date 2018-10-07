#include "rgb_to_yuv_layer.h"
#include "kernels/rgb_to_yuv_layer.cuh"

RgbToYuvLayer::RgbToYuvLayer()
              :PreprocessingLayer()
{

}

RgbToYuvLayer::RgbToYuvLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{

}

RgbToYuvLayer::~RgbToYuvLayer()
{

}



void RgbToYuvLayer::process(Tensor &output, Tensor &input)
{
  rgb_to_yuv_layer(output, input);
}
