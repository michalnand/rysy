#ifndef _GRU_LAYER_H_
#define _GRU_LAYER_H_

#include "layer.h"
#include "fc_layer.h"

class GRULayer: public Layer
{
  public:
    GRULayer();
    GRULayer(GRULayer& other);

    GRULayer(const GRULayer& other);
    virtual ~GRULayer();
    GRULayer& operator= (GRULayer& other);
    GRULayer& operator= (const GRULayer& other);

    GRULayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_gru(GRULayer &other);
    void copy_gru(const GRULayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void reset();
    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

  public:
    FCLayer *u_layer, *g_layer;
    Tensor _input, state;
    Tensor g_output, u_output;

};

#endif
