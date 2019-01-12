#ifndef _RECURRENT_LAYER_H_
#define _RECURRENT_LAYER_H_

#include "layer.h"

class RecurrentLayer: public Layer
{

  public:
    RecurrentLayer();
    RecurrentLayer(RecurrentLayer& other);

    RecurrentLayer(const RecurrentLayer& other);
    virtual ~RecurrentLayer();
    RecurrentLayer& operator= (RecurrentLayer& other);
    RecurrentLayer& operator= (const RecurrentLayer& other);

    RecurrentLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_rl(RecurrentLayer &other);
    void copy_rl(const RecurrentLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

    void reset_state();

    private:
        unsigned int max_time_window_size, time_idx;

        Tensor fc_output, fc_error;
        std::vector<Tensor> fc_input_concated, h, h_error;
};

#endif
