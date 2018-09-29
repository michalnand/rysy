#ifndef _GATING_LAYER_H_
#define _GATING_LAYER_H_

#include "layer.h"

class GatingLayer: public Layer
{
  public:
    GatingLayer();
    GatingLayer(GatingLayer& other);

    GatingLayer(const GatingLayer& other);
    virtual ~GatingLayer();
    GatingLayer& operator= (GatingLayer& other);
    GatingLayer& operator= (const GatingLayer& other);

    GatingLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_gating(GatingLayer &other);
    void copy_gating(const GatingLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
