#include "recurrent_layer.h"
#include "kernels/fc_layer.cuh"
#include "kernels/relu6_layer.cuh"

#include "kernels/w_update.cuh"

#include "../cuda_float_allocator.cuh"
#include "../cuda_tensor.cuh"

#include <iostream>

RecurrentLayer::RecurrentLayer()
        :Layer()
{

}

RecurrentLayer::RecurrentLayer(RecurrentLayer& other)
        :Layer(other)
{
  copy_rl(other);
}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& other)
        :Layer(other)
{
  copy_rl(other);
}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer& RecurrentLayer::operator= (RecurrentLayer& other)
{
  copy(other);
  copy_rl(other);
  return *this;
}

RecurrentLayer& RecurrentLayer::operator= (const RecurrentLayer& other)
{
  copy(other);
  copy_rl(other);
  return *this;
}

RecurrentLayer::RecurrentLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  unsigned int neurons_count = kernel_geometry.w*kernel_geometry.h*kernel_geometry.d;
  unsigned int inputs_count  = input_geometry.w*input_geometry.h*input_geometry.d + neurons_count;


  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = neurons_count;

  this->output_geometry.w = 1;
  this->output_geometry.h = 1;
  this->output_geometry.d = neurons_count;


  w.init(inputs_count*neurons_count);
  w_grad.init(inputs_count*neurons_count);
  bias.init(neurons_count);

    if (hyperparameters.init_weight_range > INIT_WEIGHT_RANGE_XAVIER_LIMIT)
    {
        w.set_random(hyperparameters.init_weight_range);
    }
    else
    {
        w.set_random_xavier();
    }

    bias.set_random(0.00001);

    m.init(w.get_geometry());
    v.init(w.get_geometry());


    max_time_window_size = 256;
    for (unsigned int i = 0; i < max_time_window_size; i++)
    {
        fc_input_concated.push_back(Tensor(input_geometry.w*input_geometry.h*input_geometry.d));
        h.push_back(Tensor(neurons_count));
        h_error.push_back(Tensor(neurons_count));
    }

    fc_output.init(neurons_count);
    fc_error.init(neurons_count);

    reset_state();

    time_idx = 0;

    unsigned int input_size  = input_geometry.w*input_geometry.h*input_geometry.d;
    unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
    flops = output_size*(1 + input_size)*2;

    layer_name = "RECURRENT";
}

void RecurrentLayer::copy_rl(RecurrentLayer &other)
{
  (void)other;
}

void RecurrentLayer::copy_rl(const RecurrentLayer &other)
{
  (void)other;
}



void RecurrentLayer::forward(Tensor &output, Tensor &input)
{
    //concat input for layer
    //input = X + H
    cu_device_to_device(fc_input_concated[time_idx].v, input.v, input.size());
    cu_device_to_device(fc_input_concated[time_idx].v + input.size(), h[time_idx-1].v, h[time_idx-1].size());

    //compute output
    fc_layer_forward(fc_output, fc_input_concated[time_idx], w, bias);
    relu_6_layer_forward(h[time_idx], fc_output);

    //state as output
    output.copy(h[time_idx]);

    //next time step
    if (time_idx < (max_time_window_size-1))
        time_idx++;
}

void RecurrentLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
    layer_mem_prev.error.clear();

    h_error[time_idx].add(layer_mem.error);
    for (unsigned int i = time_idx; i > 0; i--)
    {
        relu_6_layer_backward(fc_error, h[i], h_error[i]);

        fc_layer_gradient(w_grad, h[i], fc_error);
        fc_layer_update_bias(bias, fc_error, hyperparameters.learning_rate);

        /*
        //TODO error backpropagation
        fc_layer_backward(fc_error_back, fc_input_concated[i], fc_error, w);


        layer_mem_prev.error.add(error_back[0]);
        h_error[i-1].add(error_back[1]);
        */
    }

    if (update_weights)
    {
        w_update(w, w_grad, m, v, hyperparameters);
        w_grad.clear();
    }


    if (time_idx > 0)
        time_idx--;
}

void RecurrentLayer::reset_state()
{
    for (unsigned int i = 0; i < h.size(); i++)
    {
        fc_input_concated[i].clear();
        h[i].clear();
        h_error[i].clear();
    }

    fc_output.clear();
    fc_error.clear();

    time_idx = 1;
}
