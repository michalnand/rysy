#ifndef _ELU_LAYER_H_
#define _ELU_LAYER_H_

#include "layer.h"

class EluLayer: public Layer
{
  public:
    EluLayer();
    EluLayer(EluLayer& other);

    EluLayer(const EluLayer& other);
    virtual ~EluLayer();
    EluLayer& operator= (EluLayer& other);
    EluLayer& operator= (const EluLayer& other);

    EluLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_elu(EluLayer &other);
    void copy_elu(const EluLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
