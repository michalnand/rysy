#include "autoencoder_experiment.h"

#include <math.h>
#include <timer.h>
#include <image.h>

AutoencoderExperiment::AutoencoderExperiment(DatasetInterface &dataset, std::string config_dir)
{
  this->dataset    = &dataset;
  this->config_dir = config_dir;
}



AutoencoderExperiment::~AutoencoderExperiment()
{

}

void AutoencoderExperiment::run()
{
  parameters.load(config_dir+"parameters.json");

  Log experiment_log(config_dir+"experiment.log");
  Log training_progress_log(config_dir+"training_progress.log");

  experiment_log << "initializing\n";


  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  output_geometry = input_geometry;

  if (dataset->get_unlabeled_size() > 0)
    use_unlabeled = true;
  else
    use_unlabeled = false;

  if (use_unlabeled)
    experiment_log << "using UNLABELED for training with count " << dataset->get_unlabeled_size() << "\n";
  else
    experiment_log << "using TRAINING for training with count " << dataset->get_training_size() << "\n";


  CNN nn(parameters.result["network_architecture"], input_geometry, output_geometry);

    experiment_log << "training\n";

  unsigned int epoch_count = parameters.result["epoch_count"].asInt();

  unsigned int epoch_learning_rate_decay = parameters.result["epoch_learning_rate_decay"].asInt();
  float learning_rate_decay = parameters.result["learning_rate_decay"].asFloat();

  if (epoch_learning_rate_decay == 0)
    epoch_learning_rate_decay = epoch_count+1;

  float learning_rate = nn.get_learning_rate();
  float lambda1        = nn.get_lambda1();
  float lambda2        = nn.get_lambda2();

  Timer timer;

  unsigned int sub_epoch_size =  parameters.result["sub_epoch_size"].asInt();

  if (sub_epoch_size == 0)
    sub_epoch_size = 5;

  unsigned int sub_epoch_iterations = dataset->get_training_size()/sub_epoch_size;


  bool output_valid = true;


  float rms_best = input_geometry.w*input_geometry.h*input_geometry.d*1000.0;

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
        experiment_log << "testing\n";

        output_valid = test(nn);

        experiment_log << "testing done\n";

        experiment_log << "rms testing " << rms_testing_average << " " << rms_testing_min << " " << rms_testing_max << "\n";
        experiment_log << "rms training " << rms_training_average << " " << rms_training_min << " " << rms_training_max << "\n";

        float progress = epoch + sub_epoch*1.0/sub_epoch_size;

        training_progress_log << progress << " " << epoch << " " << sub_epoch << " ";
        training_progress_log << rms_testing_average << " ";
        training_progress_log << rms_testing_min << " ";
        training_progress_log << rms_testing_max << " ";
        training_progress_log << rms_training_average << " ";
        training_progress_log << rms_training_min << " ";
        training_progress_log << rms_training_max << "\n";


        if (rms_testing_average < rms_best)
        {
          rms_best = rms_testing_average;

          experiment_log << "saving best net\n";

          std::string best_net = config_dir + "trained/";
          nn.save(best_net);
          experiment_log << "best net saved to " << best_net << ", with rms " << rms_best << "%\n";

          process_best(nn);
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



void AutoencoderExperiment::train_iterations(CNN &nn, unsigned int iterations)
{
  nn.set_training_mode();

  for (unsigned int i = 0; i < iterations; i++)
  {
    sDatasetItem item;

    if (use_unlabeled)
      item = dataset->get_random_unlabeled();
    else
      item = dataset->get_random_training();

    nn.train(item.input, item.input);
  }

  nn.unset_training_mode();
}

bool AutoencoderExperiment::test(CNN &nn)
{
    float rms_max = input_geometry.w*input_geometry.h*input_geometry.d*1000.0;

    rms_testing_average = 0;
    rms_testing_min     = rms_max;
    rms_testing_max     = -rms_max;

    rms_training_average = 0;
    rms_training_min     = rms_max;
    rms_training_max     = -rms_max;

    bool valid_output = true;

    std::vector<float> nn_output(dataset->get_input_size());

    unsigned int testing_size = dataset->get_testing_size();
    for (unsigned int i = 0; i < testing_size; i++)
    {
      sDatasetItem item = dataset->get_testing(i);

      nn.forward(nn_output, item.input);

      float rms = compute_rms(item.input, nn_output);
      rms_testing_average+= rms;

      if (rms > rms_testing_max)
        rms_testing_max = rms;
      if (rms < rms_testing_min)
        rms_testing_min = rms;

      if (i < 10)
      {
        if (is_valid(nn_output) != true)
        {
          valid_output = false;
          break;
        }
      }
    }

    rms_testing_average/= testing_size;

    if (valid_output != true)
      return false;



    for (unsigned int i = 0; i < testing_size; i++)
    {
      sDatasetItem item = dataset->get_random_training();

      nn.forward(nn_output, item.input);

      float rms = compute_rms(item.input, nn_output);
      rms_training_average+= rms;

      if (rms > rms_training_max)
        rms_training_max = rms;
      if (rms < rms_training_min)
        rms_training_min = rms;

      if (i < 10)
      {
        if (is_valid(nn_output) != true)
        {
          valid_output = false;
          break;
        }
      }
    }

    rms_training_average/= testing_size;

    if (valid_output != true)
      return false;

    return true;
}


void AutoencoderExperiment::process_best(CNN &nn)
{
  std::vector<float> nn_output(dataset->get_input_size());

  for (unsigned int item_idx = 0; item_idx < 100; item_idx++)
  {
    sDatasetItem item = dataset->get_testing(item_idx%dataset->get_testing_size());

    nn.forward(nn_output, item.input);

    save_image(parameters.result["export_required_examples_path"].asString() + std::to_string(item_idx) + ".png", item.input);
    save_image(parameters.result["export_reconstructed_examples_path"].asString() + std::to_string(item_idx) + ".png", nn_output);
  }
}

bool AutoencoderExperiment::is_valid(std::vector<float> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    if (isnan(v[i]))
      return false;

  for (unsigned int i = 0; i < v.size(); i++)
    if (isinf(v[i]))
      return false;

  return true;
}

float AutoencoderExperiment::compute_rms(std::vector<float> &va, std::vector<float> &vb)
{
  float result = 0.0;
  for (unsigned int i = 0; i < va.size(); i++)
  {
    result+= (va[i] - vb[i])*(va[i] - vb[i]);
  }

  return sqrt(result/va.size());
}

void AutoencoderExperiment::save_image(std::string file_name, std::vector<float> &v)
{
  Image image(dataset->get_width(), dataset->get_height());

  if (dataset->get_channels() == 3)
    image.from_vector(v);
  else
    image.from_vector_grayscale(v);

  image.save(file_name);
}
