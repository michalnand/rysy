#include "gru_layer.h"

#include "kernels/fc_layer.cuh"
#include "kernels/gru_gate.cuh"
#include "kernels/w_update.cuh"

#include "../cuda_float_allocator.cuh"
#include "../cuda_tensor.cuh"

GRULayer::GRULayer()
        :Layer()
{

}

GRULayer::GRULayer(GRULayer& other)
        :Layer(other)
{
  copy_gru(other);
}

GRULayer::GRULayer(const GRULayer& other)
        :Layer(other)
{
  copy_gru(other);
}

GRULayer::~GRULayer()
{
  delete u_layer;
  delete g_layer;
}

GRULayer& GRULayer::operator= (GRULayer& other)
{
  copy(other);
  copy_gru(other);
  return *this;
}

GRULayer& GRULayer::operator= (const GRULayer& other)
{
  copy(other);
  copy_gru(other);
  return *this;
}

GRULayer::GRULayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  unsigned int input_size    = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int output_size   = kernel_geometry.w*kernel_geometry.h*kernel_geometry.d;


  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = output_size;

  this->output_geometry.w = 1;
  this->output_geometry.h = 1;
  this->output_geometry.d = output_size;


  sGeometry _input_geometry;
  _input_geometry.w = 1;
  _input_geometry.h = 1;
  _input_geometry.d = input_size + output_size;

  u_layer = new FCLayer(_input_geometry, kernel_geometry, hyperparameters);
  g_layer = new FCLayer(_input_geometry, kernel_geometry, hyperparameters);


  _input.init(_input_geometry);
  state.init(output_geometry);
  g_output.init(output_geometry);
  g_output.init(output_geometry);

/*
  w.init(inputs_count*neurons_count);
  w_grad.init(inputs_count*neurons_count);
  bias.init(neurons_count);

  if (hyperparameters.init_weight_range > INIT_WEIGHT_RANGE_XAVIER_LIMIT)
    w.set_random(hyperparameters.init_weight_range);
  else
    w.set_random_xavier();

  bias.set_random(0.00001);

  m.init(w.get_geometry());
  v.init(w.get_geometry());

  unsigned int input_size  = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*(1 + input_size)*2;
*/

  layer_name = "GRU";
}

void GRULayer::copy_gru(GRULayer &other)
{
  (void)other;
}

void GRULayer::copy_gru(const GRULayer &other)
{
  (void)other;
}

void GRULayer::reset()
{
  state.clear();
}

void GRULayer::forward(Tensor &output, Tensor &input)
{
  cu_device_to_device(_input.v, state.v, state.size());
  cu_device_to_device(_input.v + state.size(), input.v, input.size());

  u_layer->forward(u_output, _input);
  g_layer->forward(g_output, _input);

  gru_gate_forward(output, state, u_output, g_output);

  state.copy(output);
}

void GRULayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  /*
  fc_layer_gradient(w_grad, layer_mem_prev.output, layer_mem.error);

  fc_layer_update_bias(bias, layer_mem.error, hyperparameters.learning_rate);

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }

  fc_layer_backward(layer_mem_prev.error, layer_mem_prev.output, layer_mem.error, w);
  */
}
