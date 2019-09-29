#ifndef _UNPOOLING_LAYER_H_
#define _UNPOOLING_LAYER_H_

#include "layer.h"

class UnPoolingLayer: public Layer
{
  public:
    UnPoolingLayer();
    UnPoolingLayer(UnPoolingLayer& other);

    UnPoolingLayer(const UnPoolingLayer& other);
    virtual ~UnPoolingLayer();
    UnPoolingLayer& operator= (UnPoolingLayer& other);
    UnPoolingLayer& operator= (const UnPoolingLayer& other);

    UnPoolingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_unpooling(UnPoolingLayer &other);
    void copy_unpooling(const UnPoolingLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
