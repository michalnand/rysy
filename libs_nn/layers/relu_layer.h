#ifndef _RELU_LAYER_H_
#define _RELU_LAYER_H_

#include "layer.h"

class ReluLayer: public Layer
{
  public:
    ReluLayer();
    ReluLayer(ReluLayer& other);

    ReluLayer(const ReluLayer& other);
    virtual ~ReluLayer();
    ReluLayer& operator= (ReluLayer& other);
    ReluLayer& operator= (const ReluLayer& other);

    ReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_relu(ReluLayer &other);
    void copy_relu(const ReluLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
