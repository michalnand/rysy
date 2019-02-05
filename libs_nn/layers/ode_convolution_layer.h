#ifndef _ODE_CONVOLUTION_LAYER_H_
#define _ODE_CONVOLUTION_LAYER_H_

#include "layer.h"

class OdeConvolutionLayer: public Layer
{
  public:
    OdeConvolutionLayer();
    OdeConvolutionLayer(OdeConvolutionLayer& other);

    OdeConvolutionLayer(const OdeConvolutionLayer& other);
    virtual ~OdeConvolutionLayer();
    OdeConvolutionLayer& operator= (OdeConvolutionLayer& other);
    OdeConvolutionLayer& operator= (const OdeConvolutionLayer& other);

    OdeConvolutionLayer(sGeometry input_geometry,
                        sGeometry kernel_geometry,
                        sHyperparameters hyperparameters,
                        unsigned int iterations = 8);

  protected:
    void copy_ode_convolution(OdeConvolutionLayer &other);
    void copy_ode_convolution(const OdeConvolutionLayer &other);

  public:
    bool has_weights()
    {
      return true;
    }

    void forward(Tensor &output, Tensor &input);
    void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);

    private:
        unsigned int iterations;
        
        std::vector<Tensor> activation_out;
        std::vector<Tensor> hidden_state;
        Tensor convolution_out;
        Tensor hidden_error, convolution_error, activation_error;
};

#endif
