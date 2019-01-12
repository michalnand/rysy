#ifndef _RELU6_LAYER_H_
#define _RELU6_LAYER_H_

#include "layer.h"

class Relu6Layer: public Layer
{
  public:
    Relu6Layer();
    Relu6Layer(Relu6Layer& other);

    Relu6Layer(const Relu6Layer& other);
    virtual ~Relu6Layer();
    Relu6Layer& operator= (Relu6Layer& other);
    Relu6Layer& operator= (const Relu6Layer& other);

    Relu6Layer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_relu6(Relu6Layer &other);
    void copy_relu6(const Relu6Layer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
