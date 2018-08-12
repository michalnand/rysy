#ifndef _DENSE_FC_LAYER_H_
#define _DENSE_FC_LAYER_H_

#include "layer.h"

class DenseFCLayer: public Layer
{
  protected:
    Tensor error_fc, output_fc;

  public:
    DenseFCLayer();
    DenseFCLayer(DenseFCLayer& other);

    DenseFCLayer(const DenseFCLayer& other);
    virtual ~DenseFCLayer();
    DenseFCLayer& operator= (DenseFCLayer& other);
    DenseFCLayer& operator= (const DenseFCLayer& other);

    DenseFCLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_dense_fc(DenseFCLayer &other);
    void copy_dense_fc(const DenseFCLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
