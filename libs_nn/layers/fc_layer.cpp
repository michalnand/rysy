#include "fc_layer.h"
#include "kernels/fc_layer.cuh"
#include "kernels/w_update.cuh"

FCLayer::FCLayer()
        :Layer()
{

}

FCLayer::FCLayer(FCLayer& other)
        :Layer(other)
{
  copy_fc(other);
}

FCLayer::FCLayer(const FCLayer& other)
        :Layer(other)
{
  copy_fc(other);
}

FCLayer::~FCLayer()
{

}

FCLayer& FCLayer::operator= (FCLayer& other)
{
  copy(other);
  copy_fc(other);
  return *this;
}

FCLayer& FCLayer::operator= (const FCLayer& other)
{
  copy(other);
  copy_fc(other);
  return *this;
}

FCLayer::FCLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  unsigned int inputs_count  = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int neurons_count = kernel_geometry.w*kernel_geometry.h*kernel_geometry.d;


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
    w.set_random(hyperparameters.init_weight_range);
  else
    w.set_random_xavier();

  bias.set_random(0.00001);

  m.init(w.get_geometry());
  v.init(w.get_geometry());

  unsigned int input_size  = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int output_size = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size*(1 + input_size)*2;

  layer_name = "FC";
}

void FCLayer::copy_fc(FCLayer &other)
{
  (void)other;
}

void FCLayer::copy_fc(const FCLayer &other)
{
  (void)other;
}


void FCLayer::forward(Tensor &output, Tensor &input)
{
  fc_layer_forward(output, input, w, bias);
}

void FCLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  fc_layer_gradient(w_grad, layer_mem_prev.output, layer_mem.error);

  fc_layer_update_bias(bias, layer_mem.error, hyperparameters.learning_rate);

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }

  fc_layer_backward(layer_mem_prev.error, layer_mem_prev.output, layer_mem.error, w);
}
