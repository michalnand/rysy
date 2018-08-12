#ifndef _DROPOUT_LAYER_H_
#define _DROPOUT_LAYER_H_

#include "layer.h"

class DropoutLayer: public Layer
{
  protected:
    Tensor noise;

  public:
    DropoutLayer();
    DropoutLayer(DropoutLayer& other);

    DropoutLayer(const DropoutLayer& other);
    virtual ~DropoutLayer();
    DropoutLayer& operator= (DropoutLayer& other);
    DropoutLayer& operator= (const DropoutLayer& other);

    DropoutLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_dropout(DropoutLayer &other);
    void copy_dropout(const DropoutLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
