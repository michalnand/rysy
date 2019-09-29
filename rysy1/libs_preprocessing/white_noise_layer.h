#ifndef _WHITE_NOISE_LAYER_H_
#define _WHITE_NOISE_LAYER_H_


#include <preprocessing_layer.h>

class WhiteNoiseLayer: public PreprocessingLayer
{
  protected:
    float noise_level;

    Tensor noise;

  public:
    WhiteNoiseLayer();
    WhiteNoiseLayer(Json::Value parameters);
    virtual ~WhiteNoiseLayer();

  public:
    void process(Tensor &output, Tensor &input, unsigned int augumentation);
};


#endif
