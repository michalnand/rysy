#include "denoising_autoencoder_experiment.h"

#include <iostream>
#include <math.h>
#include <timer.h>
#include <image_save.h>


DenoisingAutoencoderExperiment::DenoisingAutoencoderExperiment(DatasetInterface &dataset, std::string config_dir)
{
  this->dataset    = &dataset;
  this->config_dir = config_dir;
}



DenoisingAutoencoderExperiment::~DenoisingAutoencoderExperiment()
{

}

void DenoisingAutoencoderExperiment::run()
{
  JsonConfig parameters(config_dir + "parameters.json");

  Log experiment_log(config_dir + "experiment.log");
  Log training_progress_log(config_dir + "training_progress.log");

  experiment_log << "initializing\n";

  sGeometry input_geometry;
  sGeometry output_geometry;


  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  output_geometry = input_geometry;

  CNN nn(parameters.result["network_architecture"], input_geometry, output_geometry);

  compare_testing.set_output_size(dataset->get_output_size());
  compare_training.set_output_size(dataset->get_output_size());


  unsigned int epoch_count = parameters.result["epoch_count"].asInt();

  unsigned int sub_epoch_size = 1;
  if (parameters.result["sub_epoch_size"] != Json::Value::null)
    sub_epoch_size = parameters.result["sub_epoch_size"].asInt();

  unsigned int epoch_learning_rate_decay = epoch_count;
  if (parameters.result["epoch_learning_rate_decay"] != Json::Value::null)
    epoch_learning_rate_decay = parameters.result["epoch_learning_rate_decay"].asInt();

  float learning_rate_decay = 1.0;
  if (parameters.result["learning_rate_decay"] != Json::Value::null)
    learning_rate_decay = parameters.result["learning_rate_decay"].asFloat();



  unsigned int sub_epoch_iterations = dataset->get_training_size()/sub_epoch_size;
  bool output_valid = true;


  float learning_rate  = nn.get_learning_rate();
  float lambda1        = nn.get_lambda1();
  float lambda2        = nn.get_lambda2();

  Timer timer;



  experiment_log << "training size    : " << dataset->get_training_size() << "\n";
  experiment_log << "testing size     : " << dataset->get_testing_size()  << "\n";
  experiment_log << "epoch count      : " << epoch_count  << "\n";
  experiment_log << "sub epoch size   : " << sub_epoch_size  << "\n";
  experiment_log << "\n";
  experiment_log << "epoch_learning_rate_decay   : " << epoch_learning_rate_decay  << "\n";
  experiment_log << "learning_rate_decay         : " << learning_rate_decay  << "\n";


  experiment_log << "\n";
  experiment_log << "training\n";

  float best_error = 1000000000.0*dataset->get_output_size();

  for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
  {
      for (unsigned int sub_epoch = 0; sub_epoch < sub_epoch_size; sub_epoch++)
      {
        experiment_log << "\n\n\n";
        experiment_log << "epoch [" << epoch << " " << sub_epoch << "] from [" << epoch_count << " " << sub_epoch_size << "]\n";

        timer.start();
        train_iterations(nn, sub_epoch_iterations);
        timer.stop();

        experiment_log << "training time per iterations " << timer.get_duration()/sub_epoch_iterations << "[ms]\n";
        experiment_log << "testing on " << dataset->get_testing_size() << " items\n";

        output_valid = test(nn);

        experiment_log << "testing done\n";

        experiment_log << "average error " << compare_testing.get_error_average_squared() << "\n";
        experiment_log << "min error     " << compare_testing.get_error_min_squared() << "\n";
        experiment_log << "max error     " << compare_testing.get_error_max_squared() << "\n";


        float progress = epoch + sub_epoch*1.0/sub_epoch_size;

        training_progress_log << progress << " " << epoch << " " << sub_epoch << " ";
        training_progress_log << compare_testing.get_error_average_squared() << " " << compare_testing.get_error_min_squared() << " " << compare_testing.get_error_max_squared() << " ";
        training_progress_log << compare_training.get_error_average_squared() << " " << compare_training.get_error_min_squared() << " " << compare_training.get_error_max_squared() << "\n";

        float error_summary = compare_testing.get_error_average_squared();

        if (best_error > error_summary)
        {
          best_error = error_summary;

          experiment_log << "saving best net\n";
          experiment_log << "error testing = " << compare_testing.get_error_average_squared() << " " << compare_testing.get_error_min_squared() << " " << compare_testing.get_error_max_squared() << "\n";
          experiment_log << "error training = " << compare_training.get_error_average_squared() << " " << compare_training.get_error_min_squared() << " " << compare_training.get_error_max_squared() << "\n";
          experiment_log << "\n";

          std::string best_net = config_dir + "trained/";
          nn.save(best_net);


           std::string examples_dir = config_dir + "examples/";
           save_examples(nn, examples_dir);

          /*
          std::string result_training_dir = config_dir + "result_training/";
          std::string result_testing_dir = config_dir + "result_testing/";

          compare_testing.save_json_file(result_testing_dir + "best_net_result.json");
          compare_testing.save_text_file(result_testing_dir);

          compare_training.save_json_file(result_training_dir + "best_net_result.json");
          compare_training.save_text_file(result_training_dir);
          */

          experiment_log << "best net saved to " << best_net << ", with error " << best_error << "\n";

          process_best();
        }

        if (output_valid != true)
          break;
      }

      if (((epoch+1)%epoch_learning_rate_decay) == 0)
      {
        learning_rate*= learning_rate_decay;
        lambda1*= learning_rate_decay;
        lambda2*= learning_rate_decay;

        nn.set_learning_rate(learning_rate);
        nn.set_lambda1(lambda1);
        nn.set_lambda2(lambda2);

        experiment_log << "setting learning rate to " << learning_rate << "\n";
      }


      if (output_valid != true)
        break;
  }

  if (output_valid != true)
  {
    experiment_log << "NaN error, ending training\n";
  }

  experiment_log << "training done\n";

}


void DenoisingAutoencoderExperiment::train_iterations(CNN &nn, unsigned int iterations)
{
  nn.set_training_mode();

  for (unsigned int i = 0; i < iterations; i++)
  {
    sDatasetItem item = dataset->get_random_training();

    nn.train(item.output, item.input);
  }

  nn.unset_training_mode();
}

bool DenoisingAutoencoderExperiment::test(CNN &nn)
{
    compare_testing.clear();
    compare_training.clear();

    compare_testing.set_output_size(dataset->get_output_size());
    compare_training.set_output_size(dataset->get_output_size());

    bool valid_output = true;

    std::vector<float> nn_output(dataset->get_output_size());

    unsigned int testing_size = dataset->get_testing_size();

    for (unsigned int i = 0; i < testing_size; i++)
    {
      sDatasetItem item = dataset->get_testing(i);

      nn.forward(nn_output, item.input);

      compare_testing.compare(item.output, nn_output);

      if (i < 10)
      {
        if (is_valid(nn_output) != true)
        {
          valid_output = false;
          break;
        }
      }
    }

    if (valid_output != true)
      return false;

    for (unsigned int i = 0; i < testing_size; i++)
    {
      sDatasetItem item = dataset->get_random_training();

      nn.forward(nn_output, item.input);

      compare_training.compare(item.output, nn_output);

      if (i < 10)
      {
        if (is_valid(nn_output) != true)
        {
          valid_output = false;
          break;
        }
      }
    }

    if (valid_output != true)
      return false;


    compare_testing.process(100);
    compare_training.process(100);

    return true;
}


void DenoisingAutoencoderExperiment::process_best()
{

}

bool DenoisingAutoencoderExperiment::is_valid(std::vector<float> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    if (isnan(v[i]))
      return false;

  for (unsigned int i = 0; i < v.size(); i++)
    if (isinf(v[i]))
      return false;

  return true;
}


void DenoisingAutoencoderExperiment::save_examples(CNN &nn, std::string path, unsigned int count)
{
    bool grayscale;

    if (dataset->get_channels() == 1)
        grayscale     = true;
    else
        grayscale     = false;

    ImageSave image(dataset->get_width(), dataset->get_height(), grayscale);


    std::vector<float> nn_output(dataset->get_output_size());


    for (unsigned int i = 0; i < count; i++)
    {
        for (unsigned int i = 0; i < nn_output.size(); i++)
            nn_output[i] = 0.0;

        sDatasetItem item = dataset->get_testing(i);
        nn.forward(nn_output, item.input);

        std::string file_name   = path + std::to_string(i) + "_2_ae_output.png";

        image.save(file_name, nn_output);
    }
}
