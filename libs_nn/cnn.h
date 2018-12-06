#ifndef _CNN_H_
#define _CNN_H_

#include "layer_memory.h"

#include "layers/layer.h"

#include <json_config.h>
#include <log.h>

class CNN
{
  public:
    CNN();

    CNN(CNN& other);
    CNN(const CNN& other);

    CNN(std::string json_file_name, sGeometry input_geometry = {0, 0, 0}, sGeometry output_geometry = {0, 0, 0}, bool detector_mode = false);
    CNN(Json::Value &json_config, sGeometry input_geometry = {0, 0, 0}, sGeometry output_geometry = {0, 0, 0}, bool detector_mode = false);

    virtual ~CNN();

    CNN& operator= (CNN& other);
    CNN& operator= (const CNN& other);

  protected:
    void copy(CNN& other);
    void copy(const CNN& other);

    void init(Json::Value &json_config, sGeometry input_geometry_, sGeometry output_geometry_, bool detector_mode = false);
    Layer* create_layer(Json::Value &parameters, sHyperparameters hyperparameters, sGeometry layer_input_geometry, bool detector_mode = false);

  public:
    void forward(std::vector<float> &output, std::vector<float> &input);
    void train(std::vector<float> &required_output, std::vector<float> &input);
    void train_single_output(float required_output, unsigned int output_idx, std::vector<float> &input);

    void forward(Tensor &output, Tensor &input);
    void train(Tensor &required_output, Tensor &input);
    void train_single_output(float required_output, unsigned int output_idx, Tensor &input);

    void train(std::vector<Tensor> &required_output, std::vector<Tensor> &input);
    void train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input);


    void set_training_mode();
    void unset_training_mode();

    void set_learning_rate(float learning_rate);
    void set_lambda1(float lambda);
    void set_lambda2(float lambda);

    float get_learning_rate()
    {
      return hyperparameters.learning_rate;
    }

    float get_lambda1()
    {
      return hyperparameters.lambda1;
    }

    float get_lambda2()
    {
      return hyperparameters.lambda2;
    }

    void save(std::string file_name_prefix);
    void load_weights(std::string file_name_prefix);

  protected:
    void forward_training(Tensor &output, Tensor &input);


  protected:
    Json::Value json_parameters;

    Log network_log;

    sHyperparameters hyperparameters;
    sGeometry input_geometry, output_geometry;

    LayerMemory input_layer_memory;
    std::vector<LayerMemory*> layer_memory;
    std::vector<Layer*> layers;

    bool training_mode;
    unsigned int minibatch_counter;

  protected:
    Tensor nn_input, nn_output, nn_required_output;


};

#endif
