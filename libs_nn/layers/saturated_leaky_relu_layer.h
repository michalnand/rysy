#ifndef _SATURATED_LEAKY_RELU_LAYER_H_
#define _SATURATED_LEAKY_RELU_LAYER_H_

#include "layer.h"

class SaturatedLeakyReluLayer: public Layer
{
  public:
    SaturatedLeakyReluLayer();
    SaturatedLeakyReluLayer(SaturatedLeakyReluLayer& other);

    SaturatedLeakyReluLayer(const SaturatedLeakyReluLayer& other);
    virtual ~SaturatedLeakyReluLayer();
    SaturatedLeakyReluLayer& operator= (SaturatedLeakyReluLayer& other);
    SaturatedLeakyReluLayer& operator= (const SaturatedLeakyReluLayer& other);

    SaturatedLeakyReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_saturated_leaky_relu(SaturatedLeakyReluLayer &other);
    void copy_saturated_leaky_relu(const SaturatedLeakyReluLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
