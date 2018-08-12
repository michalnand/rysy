#include "dense_fc_layer.h"

#include "kernels/fc_layer.cuh"
#include "kernels/w_update.cuh"

#include "../cuda_float_allocator.cuh"
#include "../cuda_tensor.cuh"



DenseFCLayer::DenseFCLayer()
        :Layer()
{

}

DenseFCLayer::DenseFCLayer(DenseFCLayer& other)
        :Layer(other)
{
  copy_dense_fc(other);
}

DenseFCLayer::DenseFCLayer(const DenseFCLayer& other)
        :Layer(other)
{
  copy_dense_fc(other);
}

DenseFCLayer::~DenseFCLayer()
{

}

DenseFCLayer& DenseFCLayer::operator= (DenseFCLayer& other)
{
  copy(other);
  copy_dense_fc(other);
  return *this;
}

DenseFCLayer& DenseFCLayer::operator= (const DenseFCLayer& other)
{
  copy(other);
  copy_dense_fc(other);
  return *this;
}

DenseFCLayer::DenseFCLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  unsigned int inputs_count  = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int neurons_count = kernel_geometry.w*kernel_geometry.h*kernel_geometry.d;


  this->kernel_geometry.w = 1;
  this->kernel_geometry.h = 1;
  this->kernel_geometry.d = neurons_count;

  this->output_geometry.w = 1;
  this->output_geometry.h = 1;
  this->output_geometry.d = neurons_count + inputs_count;

  sGeometry fc_output_geometry;
  fc_output_geometry.w = 1;
  fc_output_geometry.h = 1;
  fc_output_geometry.d = neurons_count;

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

  error_fc.init(fc_output_geometry);
  output_fc.init(fc_output_geometry);

  unsigned int input_size  = input_geometry.w*input_geometry.h*input_geometry.d;
  unsigned int output_size = fc_output_geometry.w*fc_output_geometry.h*fc_output_geometry.d;
  flops = output_size*(1 + input_size)*2;

  layer_name = "DENSE FC";
}

void DenseFCLayer::copy_dense_fc(DenseFCLayer &other)
{
  error_fc  = other.error_fc;
  output_fc = other.output_fc;
}

void DenseFCLayer::copy_dense_fc(const DenseFCLayer &other)
{
  error_fc  = other.error_fc;
  output_fc = other.output_fc;
}


void DenseFCLayer::forward(Tensor &output, Tensor &input)
{
  fc_layer_forward(output_fc, input, w, bias);

  cuda_float_allocator.device_to_device(output.v, output_fc.v, output_fc.size());
  cuda_float_allocator.device_to_device(output.v + output_fc.size(), input.v, input.size());
}

void DenseFCLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  cuda_float_allocator.device_to_device(error_fc.v, layer_mem.error.v, error_fc.size());

  fc_layer_gradient(w_grad, layer_mem_prev.output, error_fc);

  fc_layer_update_bias(bias, error_fc, hyperparameters.learning_rate);

  if (update_weights)
  {
    w_update(w, w_grad, m, v, hyperparameters);
    w_grad.clear();
  }

  fc_layer_backward(layer_mem_prev.error, layer_mem_prev.output, error_fc, w);
  cuda_tensor_add(layer_mem_prev.error.v, layer_mem.error.v + error_fc.size(), layer_mem_prev.error.size());

}
