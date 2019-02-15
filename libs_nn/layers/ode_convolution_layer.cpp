#include "ode_convolution_layer.h"

#include "kernels/convolution_layer_forward.cuh"
#include "kernels/convolution_layer_backward.cuh"
#include "kernels/leaky_relu_layer.cuh"

#include "kernels/w_update.cuh"


OdeConvolutionLayer::OdeConvolutionLayer()
        :Layer()
{

}

OdeConvolutionLayer::OdeConvolutionLayer(OdeConvolutionLayer& other)
        :Layer(other)
{
  copy_ode_convolution(other);
}

OdeConvolutionLayer::OdeConvolutionLayer(const OdeConvolutionLayer& other)
        :Layer(other)
{
  copy_ode_convolution(other);
}

OdeConvolutionLayer::~OdeConvolutionLayer()
{

}

OdeConvolutionLayer& OdeConvolutionLayer::operator= (OdeConvolutionLayer& other)
{
  copy(other);
  copy_ode_convolution(other);
  return *this;
}

OdeConvolutionLayer& OdeConvolutionLayer::operator= (const OdeConvolutionLayer& other)
{
  copy(other);
  copy_ode_convolution(other);
  return *this;
}

OdeConvolutionLayer::OdeConvolutionLayer(   sGeometry input_geometry,
                                            sGeometry kernel_geometry,
                                            sHyperparameters hyperparameters,
                                            unsigned int iterations)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
    this->input_geometry.w = input_geometry.w;
    this->input_geometry.h = input_geometry.h;
    this->input_geometry.d = input_geometry.d;

    this->kernel_geometry.d = input_geometry.d;

    this->output_geometry.w = this->input_geometry.w;
    this->output_geometry.h = this->input_geometry.h;
    this->output_geometry.d = this->kernel_geometry.d;

    this->iterations = iterations;

    sGeometry tmp_kernel_geometry = kernel_geometry;
    tmp_kernel_geometry.d = this->input_geometry.d*this->output_geometry.d;

    w.init(tmp_kernel_geometry);

    w.set_random_xavier();

    w_grad.init(tmp_kernel_geometry);

    bias.init(this->output_geometry.d);
    bias.set_random(0.0000001);

    m.init(w.get_geometry());
    v.init(w.get_geometry());



    for (unsigned int i = 0; i < iterations + 1; i++)
      h.push_back(Tensor(this->output_geometry));

    for (unsigned int i = 0; i < iterations; i++)
      g.push_back(Tensor(this->output_geometry));

    for (unsigned int i = 0; i < iterations; i++)
      f.push_back(Tensor(this->output_geometry));

    e.init(output_geometry);
    ef.init(output_geometry);
    eg.init(output_geometry);

    unsigned int output_size  = output_geometry.w*output_geometry.h;
    flops = output_size*kernel_geometry.d*(1 + kernel_geometry.w*kernel_geometry.h*input_geometry.d); //convolution flops
    flops+= 2*output_size;    //activation flops
    flops = flops*this->iterations;

    layer_name = "ODE CONV";
}

void OdeConvolutionLayer::copy_ode_convolution(OdeConvolutionLayer &other)
{
  (void)other;
}

void OdeConvolutionLayer::copy_ode_convolution(const OdeConvolutionLayer &other)
{
  (void)other;
}


void OdeConvolutionLayer::forward(Tensor &output, Tensor &input)
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    h[0].copy(input);

    for (unsigned int i = 0; i < iterations; i++)
    {
        convolution_layer_forward(g[i], h[i], w, bias);
        leaky_relu_layer_forward(f[i], g[i]);

        h[i + 1].copy(f[i]);  //copy layer output
        h[i + 1].add(h[i]);   //add residuum
    }

    output.copy(h[iterations]);
}


void OdeConvolutionLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  e.copy(layer_mem.error);

  int i = iterations - 1;
  while (i >= 0)
  {
    //activation error backpropagate
    leaky_relu_layer_backward(ef, f[i], e);

    //convolution layer compute gradient
    convolution_layer_gradient(w_grad, h[i], ef);
    convolution_layer_update_bias(bias, ef, hyperparameters.learning_rate);

    //backpropagate error
    convolution_layer_backward(eg, h[i], ef, w);

    //sum error
    e.add(eg);

    i--;
  }

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }

  layer_mem_prev.error.copy(e);
}
