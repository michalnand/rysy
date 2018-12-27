#ifndef _AUGMENTATION_LAYER_H_
#define _AUGMENTATION_LAYER_H_

#include <preprocessing_layer.h>

class AugmentationLayer: public PreprocessingLayer
{
  public:
    AugmentationLayer();
    AugmentationLayer(Json::Value parameters);
    virtual ~AugmentationLayer();

  public:
    void process(Tensor &output, Tensor &input, unsigned int augmentation);
};

 
#endif
