#ifndef _RGB_TO_YUV_LAYER_H_
#define _RGB_TO_YUV_LAYER_H_


#include <preprocessing_layer.h>

class RgbToYuvLayer: public PreprocessingLayer
{
  protected:
    Json::Value parameters;

  public:
    RgbToYuvLayer();
    RgbToYuvLayer(Json::Value parameters);
    virtual ~RgbToYuvLayer();

  public:
    void process(Tensor &output, Tensor &input);
};


#endif
