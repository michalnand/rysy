#ifndef _NOISE_LAYER_H_
#define _NOISE_LAYER_H_

#include "layer.h"

class NoiseLayer: public Layer
{
  protected:
    Tensor white_noise;
    Tensor salt_and_pepper_noise;

  public:
    NoiseLayer();
    NoiseLayer(NoiseLayer& other);

    NoiseLayer(const NoiseLayer& other);
    virtual ~NoiseLayer();
    NoiseLayer& operator= (NoiseLayer& other);
    NoiseLayer& operator= (const NoiseLayer& other);

    NoiseLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_noise(NoiseLayer &other);
    void copy_noise(const NoiseLayer &other);

  public:

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);
};

#endif
