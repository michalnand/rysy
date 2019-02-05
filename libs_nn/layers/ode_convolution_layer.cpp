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

    for (unsigned int i = 0; i < iterations; i++)
        activation_out.push_back(Tensor(this->output_geometry));

    for (unsigned int i = 0; i < (iterations+1); i++)
        hidden_state.push_back(Tensor(this->output_geometry));

    convolution_out.init(this->output_geometry);
    hidden_error.init(this->output_geometry);
    convolution_error.init(this->output_geometry);
    activation_error.init(this->output_geometry);


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
    for (unsigned int i = 0; i < hidden_state.size(); i++)
        hidden_state[i].clear();

    hidden_state[0].copy(input);

    for (unsigned int i = 0; i < iterations; i++)
    {
        convolution_layer_forward(convolution_out, hidden_state[i], w, bias);
        leaky_relu_layer_forward(activation_out[i], convolution_out);
        hidden_state[i+1].copy(hidden_state[i]);
        hidden_state[i+1].add(activation_out[i]);
    }

    output.copy(hidden_state[iterations]);
}


void OdeConvolutionLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
    hidden_error.copy(layer_mem.error);

    for (int i = (iterations-1); i >= 0; i--)
    {
        leaky_relu_layer_backward(activation_error, activation_out[i], hidden_error);

        convolution_layer_gradient(w_grad, hidden_state[i], activation_error);
        convolution_layer_update_bias(bias, activation_error, hyperparameters.learning_rate);

        //backpropagate error
        convolution_layer_backward(convolution_error, hidden_state[i], activation_error, w);

        hidden_error.add(convolution_error);
    }


    if (update_weights)
    {
        w_update(w, w_grad, m, v, hyperparameters);
        w_grad.clear();
    }

    layer_mem_prev.error.copy(hidden_error);
}
