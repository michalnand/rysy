#ifndef _AVERAGE_POOLING_LAYER_H_
#define _AVERAGE_POOLING_LAYER_H_

#include "layer.h"

class AveragePoolingLayer: public Layer
{
  public:
    AveragePoolingLayer();
    AveragePoolingLayer(AveragePoolingLayer& other);

    AveragePoolingLayer(const AveragePoolingLayer& other);
    virtual ~AveragePoolingLayer();
    AveragePoolingLayer& operator= (AveragePoolingLayer& other);
    AveragePoolingLayer& operator= (const AveragePoolingLayer& other);

    AveragePoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_max_pooling(AveragePoolingLayer &other);
    void copy_max_pooling(const AveragePoolingLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
