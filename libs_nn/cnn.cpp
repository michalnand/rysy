#include "cnn.h"
#include "svg_visualiser.h"
#include <image.h>

#include "layers/relu_layer.h"
#include "layers/fc_layer.h"
#include "layers/dense_fc_layer.h"
#include "layers/convolution_layer.h"
#include "layers/dense_convolution_layer.h"

#include "layers/dropout_layer.h"
#include "layers/max_pooling_layer.h"
#include "layers/unpooling_layer.h"

CNN::CNN()
{

}

CNN::CNN(CNN& other)
{
  copy(other);
}

CNN::CNN(const CNN& other)
{
  copy(other);
}

CNN::CNN(std::string json_file_name, sGeometry input_geometry, sGeometry output_geometry)
{
  JsonConfig json(json_file_name);
  init(json.result, input_geometry, output_geometry);
}

CNN::CNN(Json::Value &json_config, sGeometry input_geometry, sGeometry output_geometry)
{
  init(json_config, input_geometry, output_geometry);
}


CNN::~CNN()
{
  for (unsigned int i = 0; i < layers.size(); i++)
  {
    delete layers[i];
    layers[i] = nullptr;
  }

  for (unsigned int i = 0; i < layer_memory.size(); i++)
  {
    delete layer_memory[i];
    layer_memory[i] = nullptr;
  }

  network_log << "network destructor done\n";
}

CNN& CNN::operator= (CNN& other)
{
  copy(other);

  return *this;
}

CNN& CNN::operator= (const CNN& other)
{
  copy(other);

  return *this;
}

void CNN::copy(CNN& other)
{
  input_geometry  = other.input_geometry;
  output_geometry = other.output_geometry;
  json_parameters = other.json_parameters;

  layers          = other.layers;
}

void CNN::copy(const CNN& other)
{
  input_geometry  = other.input_geometry;
  output_geometry = other.output_geometry;
  json_parameters = other.json_parameters;
  layers          = other.layers;
}


void CNN::init(Json::Value &json_config, sGeometry input_geometry_, sGeometry output_geometry_)
{

  json_parameters = json_config;

  std::string network_log_file_name = json_parameters["network_log_file_name"].asString();

  if (network_log_file_name.size() > 0)
    network_log.set_output_file_name(network_log_file_name);

  network_log << "network init start\n";

  hyperparameters.init_weight_range = json_config["hyperparameters"]["init_weight_range"].asFloat();
  hyperparameters.learning_rate     = json_config["hyperparameters"]["learning_rate"].asFloat();
  hyperparameters.lambda            = json_config["hyperparameters"]["lambda"].asFloat();
  hyperparameters.dropout           = json_config["hyperparameters"]["dropout"].asFloat();
  hyperparameters.noise             = json_config["hyperparameters"]["noise"].asFloat();

  hyperparameters.beta1   = 0.9;
  hyperparameters.beta2   = 0.999;
  hyperparameters.epsilon = 0.00000001;

  hyperparameters.minibatch_size             = json_config["hyperparameters"]["minibatch_size"].asInt();


  if ((input_geometry_.w == 0) ||
      (input_geometry_.h == 0) ||
      (input_geometry_.d == 0) )
  {
    input_geometry.w  = json_config["input_geometry"][0].asInt();
    input_geometry.h  = json_config["input_geometry"][1].asInt();
    input_geometry.d  = json_config["input_geometry"][2].asInt();
  }
  else
  {
    input_geometry.w  = input_geometry_.w;
    input_geometry.h  = input_geometry_.h;
    input_geometry.d  = input_geometry_.d;
  }


  if ((output_geometry_.w == 0) ||
      (output_geometry_.h == 0) ||
      (output_geometry_.d == 0) )
  {
    output_geometry.w  = json_config["output_geometry"][0].asInt();
    output_geometry.h  = json_config["output_geometry"][1].asInt();
    output_geometry.d  = json_config["output_geometry"][2].asInt();
  }
  else
  {
    output_geometry.w  = output_geometry_.w;
    output_geometry.h  = output_geometry_.h;
    output_geometry.d  = output_geometry_.d;
  }


  network_log << "\nhyperparameters\n";

  network_log << "init_weight_range " << hyperparameters.init_weight_range;

  if (hyperparameters.init_weight_range < INIT_WEIGHT_RANGE_XAVIER_LIMIT)
    network_log << " ,xavier\n";
  else
    network_log << " ,uniform\n";


  network_log << "learning_rate " << hyperparameters.learning_rate << "\n";
  network_log << "lambda " << hyperparameters.lambda << "\n";
  network_log << "dropout " << hyperparameters.dropout << "\n";
  network_log << "noise " << hyperparameters.noise << "\n";
  network_log << "minibatch size " << hyperparameters.minibatch_size << "\n";

  network_log << "\n";

  input_layer_memory.init(input_geometry);

  sGeometry layer_input_geometry    = input_geometry;

  for (auto layer: json_config["layers"])
  {
    Layer *layer_ = create_layer(layer, hyperparameters, layer_input_geometry);

    if (layer_ != nullptr)
    {
      layers.push_back(layer_);
      layer_memory.push_back(new LayerMemory(layer_->get_output_geometry()));

      layer_input_geometry = layer_->get_output_geometry();
    }
    else
    {
      network_log << "ERROR, ending initialization\n";
      break;
    }
  }

  output_geometry = layer_input_geometry;

  nn_input.init(input_geometry);
  nn_output.init(output_geometry);
  nn_required_output.init(output_geometry);

  unset_training_mode();


  network_log << "\n";
  network_log << "input_geometry  [" << input_geometry.w << " " << input_geometry.h << " "  << input_geometry.d << "]\n";
  network_log << "output_geometry [" << output_geometry.w << " " << output_geometry.h << " "  << output_geometry.d << "]\n";

  unsigned long int flops = 0;

  for (unsigned int i = 0; i < layers.size(); i++)
    flops+= layers[i]->get_flops();

  network_log << "network flops operations " << flops << " FLOPS, " << flops/1000000000.0 << " GFLOPS\n";


  network_log << "init DONE\n\n";
}





Layer* CNN::create_layer(Json::Value &parameters, sHyperparameters hyperparameters, sGeometry layer_input_geometry)
{
  Layer *result = nullptr;

  std::string type = parameters["type"].asString();

  sGeometry layer_kernel_geometry;


  if (type == "relu")
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new ReluLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }


  if (type == "fc")
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new FCLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if ((type == "dense fc")||(type == "dense_fc"))
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new DenseFCLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if (type == "convolution")
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new ConvolutionLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if ((type == "dense convolution")||(type == "dense_convolution"))
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new DenseConvolutionLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }


  if (type == "output")
  {
    layer_kernel_geometry.w = 1;
    layer_kernel_geometry.h = 1;
    layer_kernel_geometry.d = output_geometry.d;
    result = new FCLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if (type == "dropout")
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new DropoutLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if ((type == "max pooling")||(type == "max_pooling"))
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new MaxPoolingLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if (type == "unpooling")
  {
    layer_kernel_geometry.w = parameters["geometry"][0].asInt();
    layer_kernel_geometry.h = parameters["geometry"][1].asInt();
    layer_kernel_geometry.d = parameters["geometry"][2].asInt();
    result = new UnPoolingLayer(layer_input_geometry, layer_kernel_geometry, hyperparameters);
  }

  if (result == nullptr)
  {
    network_log << "unknow layer : " << type << "\n";
  }
  else
  {
    std::string layer_info = result->get_info();
    network_log << layer_info << "\n";
  }

  if (result != nullptr)
  if (result->has_weights())
  {
    std::string weights_file_name_prefix = parameters["weights_file_name"].asString();
    if (weights_file_name_prefix.size() > 0)
      result->load(weights_file_name_prefix);
  }

  return result;
}


void CNN::forward(std::vector<float> &output, std::vector<float> &input)
{
  nn_input.set_from_host(input);
  forward(nn_output, nn_input);
  nn_output.set_to_host(output);
}

void CNN::train(std::vector<float> &required_output, std::vector<float> &input)
{
  nn_input.set_from_host(input);
  nn_required_output.set_from_host(required_output);

  train(nn_required_output, nn_input);
}

void CNN::forward(Tensor &output, Tensor &input)
{
  for (unsigned int i = 0; i < layers.size(); i++)
  {
    if (i == 0)
      layers[i]->forward(layer_memory[i]->output, input);

    else
      layers[i]->forward(layer_memory[i]->output, layer_memory[i-1]->output);
  }

  output.copy(layer_memory[layers.size()-1]->output);
}

void CNN::forward_training(Tensor &output, Tensor &input)
{
  input_layer_memory.output.copy(input);

  for (unsigned int i = 0; i < layers.size(); i++)
  {
    if (i == 0)
      layers[i]->forward(layer_memory[i]->output, input_layer_memory.output);
    else
      layers[i]->forward(layer_memory[i]->output, layer_memory[i-1]->output);
  }

  output.copy(layer_memory[layers.size()-1]->output);
}


void CNN::train(Tensor &required_output, Tensor &input)
{
  forward_training(nn_output, input);

  unsigned int last_idx = layers.size()-1;

  layer_memory[last_idx]->error.copy(required_output);
  layer_memory[last_idx]->error.sub(nn_output);

  bool update_weights = false;
  minibatch_counter++;

  if (minibatch_counter >= hyperparameters.minibatch_size)
  {
    update_weights = true;
    minibatch_counter = 0;
  }

  for (int i = last_idx; i>= 0; i--)
  {
    if (i == 0)
      layers[i]->backward(input_layer_memory, *layer_memory[i], update_weights);
    else
      layers[i]->backward(*layer_memory[i-1], *layer_memory[i], update_weights);
  }
}


void CNN::set_training_mode()
{
  training_mode     = true;
  minibatch_counter = 0;

  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->set_training_mode();

  input_layer_memory.clear();

  for (unsigned int i = 0; i < layer_memory.size(); i++)
    layer_memory[i]->clear();
}

void CNN::unset_training_mode()
{
  training_mode = false;
  minibatch_counter = 0;

  for (unsigned int i = 0; i < layer_memory.size(); i++)
    layers[i]->unset_training_mode();

  input_layer_memory.clear();

  for (unsigned int i = 0; i < layer_memory.size(); i++)
    layer_memory[i]->clear();
}

void CNN::set_learning_rate(float learning_rate)
{
  hyperparameters.learning_rate = learning_rate;
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->set_learning_rate(hyperparameters.learning_rate);
}

void CNN::set_lambda(float lambda)
{
  hyperparameters.lambda = lambda;
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->set_learning_rate(hyperparameters.lambda);
}

void CNN::save(std::string file_name_prefix)
{
  JsonConfig json;

  // network_log << "saving network to " << file_name_prefix << "\n";

  json_parameters["hyperparameters"]["init_weight_range"] = hyperparameters.init_weight_range;
  json_parameters["hyperparameters"]["learning_rate"]     = hyperparameters.learning_rate;
  json_parameters["hyperparameters"]["lambda"]            = hyperparameters.lambda;
  json_parameters["hyperparameters"]["dropout"]           = hyperparameters.dropout;
  json_parameters["hyperparameters"]["noise"]             = hyperparameters.noise;
  json_parameters["hyperparameters"]["minibatch_size"]    = hyperparameters.minibatch_size;


  json.result = json_parameters;

  for (unsigned int layer = 0; layer < layers.size(); layer++)
  {
    std::string weights_file_name;
    std::string weights_image_file_name;

    weights_file_name = file_name_prefix + "layer_" + std::to_string(layer);
    json.result["layers"][layer]["weights_file_name"] = weights_file_name;

    layers[layer]->save(weights_file_name);
  }


  json.save(file_name_prefix + "cnn_config.json");

  SVGVisualiser svg(json.result);
  svg.process(file_name_prefix + "cnn_architecture.svg");

  Image image(file_name_prefix + "cnn_architecture.svg");
  image.save(file_name_prefix + "cnn_architecture.png");

  // network_log << "saving done\n";
}
