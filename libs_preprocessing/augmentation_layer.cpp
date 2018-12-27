#include "augmentation_layer.h"
#include "kernels/augmentation_layer.cuh"

#include <iostream>

AugmentationLayer::AugmentationLayer()
              :PreprocessingLayer()
{
    
}

AugmentationLayer::AugmentationLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{

}

AugmentationLayer::~AugmentationLayer()
{

}



void AugmentationLayer::process(Tensor &output, Tensor &input, unsigned int augmentation)
{
  augmentation = augmentation%8;

  augmentation_layer(output, input, augmentation);
}
