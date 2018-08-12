#ifndef _MAX_POOLING_LAYER_H_
#define _MAX_POOLING_LAYER_H_

#include "layer.h"

class MaxPoolingLayer: public Layer
{
  private:
    Tensor max_mask;

  public:
    MaxPoolingLayer();
    MaxPoolingLayer(MaxPoolingLayer& other);

    MaxPoolingLayer(const MaxPoolingLayer& other);
    virtual ~MaxPoolingLayer();
    MaxPoolingLayer& operator= (MaxPoolingLayer& other);
    MaxPoolingLayer& operator= (const MaxPoolingLayer& other);

    MaxPoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_max_pooling(MaxPoolingLayer &other);
    void copy_max_pooling(const MaxPoolingLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
