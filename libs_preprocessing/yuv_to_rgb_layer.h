#ifndef _YUV_TO_RGB_LAYER_H_
#define _YUV_TO_RGB_LAYER_H_


#include <preprocessing_layer.h>

class YuvToRgbLayer: public PreprocessingLayer
{
  protected:
    Json::Value parameters;

  public:
    YuvToRgbLayer();
    YuvToRgbLayer(Json::Value parameters);
    virtual ~YuvToRgbLayer();

  public:
    void process(Tensor &output, Tensor &input);
};


#endif
