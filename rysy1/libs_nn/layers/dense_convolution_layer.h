#ifndef _DENSE_CONVOLUTION_LAYER_H_
#define _DENSE_CONVOLUTION_LAYER_H_

#include "layer.h"

class DenseConvolutionLayer: public Layer
{
  protected:
    Tensor error_convolution, output_convolution;

  public:
    DenseConvolutionLayer();
    DenseConvolutionLayer(DenseConvolutionLayer& other);

    DenseConvolutionLayer(const DenseConvolutionLayer& other);
    virtual ~DenseConvolutionLayer();
    DenseConvolutionLayer& operator= (DenseConvolutionLayer& other);
    DenseConvolutionLayer& operator= (const DenseConvolutionLayer& other);

    DenseConvolutionLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy_dense_convolution(DenseConvolutionLayer &other);
    void copy_dense_convolution(const DenseConvolutionLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

};

#endif
