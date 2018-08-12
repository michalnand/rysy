#include "layer.h"

#include <sstream>
#include <iomanip>
#include <iostream>

Layer::Layer()
{
  input_geometry.w = 0;
  input_geometry.h = 0;
  input_geometry.d = 0;

  kernel_geometry.w = 0;
  kernel_geometry.h = 0;
  kernel_geometry.d = 0;

  output_geometry.w = 0;
  output_geometry.h = 0;
  output_geometry.d = 0;

  flops = 0;
  layer_name = "ILAYER";
}

Layer::Layer(Layer& other)
{
  copy(other);
}

Layer::Layer(const Layer& other)
{
  copy(other);
}

Layer::~Layer()
{

}

Layer& Layer::operator= (Layer& other)
{
  copy(other);
  return *this;
}

Layer& Layer::operator= (const Layer& other)
{
  copy(other);
  return *this;
}

Layer::Layer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
{
  this->input_geometry  = input_geometry;
  this->kernel_geometry = kernel_geometry;
  this->hyperparameters = hyperparameters;

  flops = 0;
}

void Layer::copy(Layer &other)
{
  input_geometry  = other.input_geometry;
  kernel_geometry = other.kernel_geometry;
  output_geometry = other.output_geometry;
  hyperparameters = other.hyperparameters;

  w = other.w;
  bias = other.bias;
  m = other.m;
  v = other.v;

  training_mode = other.training_mode;

  flops = other.flops;

  layer_name    = other.layer_name;
}

void Layer::copy(const Layer &other)
{
  input_geometry  = other.input_geometry;
  kernel_geometry = other.kernel_geometry;
  output_geometry = other.output_geometry;
  hyperparameters = other.hyperparameters;

  w     = other.w;
  bias  = other.bias;
  m     = other.m;
  v     = other.v;

  training_mode = other.training_mode;

  flops = other.flops;

  layer_name    = other.layer_name;
}


void Layer::set_training_mode()
{
  training_mode = true;
}

void Layer::unset_training_mode()
{
  training_mode = false;
}

bool Layer::is_training_mode()
{
  return training_mode;
}


void Layer::save(std::string file_name_prefix)
{
  if ((w.size() != 0) || (bias.size() != 0))
  {
    std::string weight_file_name  = file_name_prefix + "_weight.bin";
    std::string bias_file_name    = file_name_prefix + "_bias.bin";

    w.save_to_file(weight_file_name);
    bias.save_to_file(bias_file_name);
  }
}

void Layer::load(std::string file_name_prefix)
{
  if ((w.size() != 0) || (bias.size() != 0))
  {
    std::string weight_file_name  = file_name_prefix + "_weight.bin";
    std::string bias_file_name    = file_name_prefix + "_bias.bin";

    w.load_from_file(weight_file_name);
    bias.load_from_file(bias_file_name);
  }
}

std::string Layer::get_info()
{
    std::stringstream str_stream;

    str_stream << std::setw(20) << layer_name << std::setw(2) << " ";
    str_stream << "[" << std::setw(5) << input_geometry.w << " " << std::setw(5) << input_geometry.h << " " << std::setw(5) << input_geometry.d << "] ";
    str_stream << "[" << std::setw(5) << kernel_geometry.w << " " << std::setw(5) << kernel_geometry.h << " " << std::setw(5) << kernel_geometry.d << "] ";
    str_stream << "[" << std::setw(5) << output_geometry.w << " " << std::setw(5) << output_geometry.h << " " << std::setw(5) << output_geometry.d << "] ";
    str_stream << "[" << std::setw(10) << w.size() << " " << std::setw(10) << bias.size() << "]";
    str_stream << "[" << std::setw(10) << flops << "]";

    return str_stream.str();
}

void Layer::print()
{
  std::cout << get_info() << "\n";
}

unsigned int Layer::get_required_memory()
{
  unsigned int result = 0;

  result+= w.size();
  result+= bias.size();
  result+= m.size();
  result+= v.size();

  return sizeof(float)*result;
}

void Layer::set_learning_rate(float learning_rate)
{
  hyperparameters.learning_rate = learning_rate;
}

void Layer::set_lambda(float lambda)
{
  hyperparameters.lambda = lambda;
}

void Layer::forward(Tensor &output, Tensor &input)
{
  (void)output;
  (void)input;
}

void Layer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  (void)update_weights;
  (void)layer_mem_prev;
  (void)layer_mem;
}
