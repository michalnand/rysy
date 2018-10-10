#ifndef _LUMA_NOISE_LAYER_H_
#define _LUMA_NOISE_LAYER_H_


#include <preprocessing_layer.h>

class LumaNoiseLayer: public PreprocessingLayer
{
  protected:
    float noise_level;

  public:
    LumaNoiseLayer();
    LumaNoiseLayer(Json::Value parameters);
    virtual ~LumaNoiseLayer();

  public:
    void process(Tensor &output, Tensor &input, unsigned int augumentation);
};


#endif
