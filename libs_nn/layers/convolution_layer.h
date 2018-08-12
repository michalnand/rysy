#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_

#include "layer.h"

class ConvolutionLayer: public Layer
{
  public:
    ConvolutionLayer();
    ConvolutionLayer(ConvolutionLayer& other);

    ConvolutionLayer(const ConvolutionLayer& other);
    virtual ~ConvolutionLayer();
    ConvolutionLayer& operator= (ConvolutionLayer& other);
    ConvolutionLayer& operator= (const ConvolutionLayer& other);

    ConvolutionLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_convolution(ConvolutionLayer &other);
    void copy_convolution(const ConvolutionLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
