#include "yuv_to_rgb_layer.h"
#include "kernels/yuv_to_rgb_layer.cuh"

YuvToRgbLayer::YuvToRgbLayer()
              :PreprocessingLayer()
{

}

YuvToRgbLayer::YuvToRgbLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{

}

YuvToRgbLayer::~YuvToRgbLayer()
{

}



void YuvToRgbLayer::process(Tensor &output, Tensor &input)
{
  yuv_to_rgb_layer(output, input);
}
