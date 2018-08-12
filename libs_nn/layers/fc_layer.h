#ifndef _FC_LAYER_H_
#define _FC_LAYER_H_

#include "layer.h"

class FCLayer: public Layer
{
  public:
    FCLayer();
    FCLayer(FCLayer& other);

    FCLayer(const FCLayer& other);
    virtual ~FCLayer();
    FCLayer& operator= (FCLayer& other);
    FCLayer& operator= (const FCLayer& other);

    FCLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_fc(FCLayer &other);
    void copy_fc(const FCLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
