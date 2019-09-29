#ifndef _LEAKY_RELU_LAYER_H_
#define _LEAKY_RELU_LAYER_H_

#include "layer.h"

class LeakyReluLayer: public Layer
{
  public:
    LeakyReluLayer();
    LeakyReluLayer(LeakyReluLayer& other);

    LeakyReluLayer(const LeakyReluLayer& other);
    virtual ~LeakyReluLayer();
    LeakyReluLayer& operator= (LeakyReluLayer& other);
    LeakyReluLayer& operator= (const LeakyReluLayer& other);

    LeakyReluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_leaky_relu(LeakyReluLayer &other);
    void copy_leaky_relu(const LeakyReluLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
