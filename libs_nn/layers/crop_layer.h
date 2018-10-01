#ifndef _CROP_LAYER_H_
#define _CROP_LAYER_H_

#include "layer.h"

class CropLayer: public Layer
{
  public:
    CropLayer();
    CropLayer(CropLayer& other);

    CropLayer(const CropLayer& other);
    virtual ~CropLayer();
    CropLayer& operator= (CropLayer& other);
    CropLayer& operator= (const CropLayer& other);

    CropLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_crop(CropLayer &other);
    void copy_crop(const CropLayer &other);

  public:
    bool has_weights()
    {
      return false;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
