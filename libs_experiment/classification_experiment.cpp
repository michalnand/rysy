#include "classification_experiment.h"

#include <math.h>
#include <classification_compare.h>
#include <timer.h>


ClassificationExperiment::ClassificationExperiment(DatasetInterface &dataset, std::string config_dir)
{
  this->dataset    = &dataset;
  this->config_dir = config_dir;
}



ClassificationExperiment::~ClassificationExperiment()
{

}

void ClassificationExperiment::run()
{
  JsonConfig parameters(config_dir+"parameters.json");

  Log experiment_log(config_dir+"experiment.log");
  Log training_progress_log(config_dir+"training_progress.log");

  experiment_log << "initializing\n";

  sGeometry input_geometry;
  sGeometry output_geometry;


  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset->get_output_size();

  CNN nn(parameters.result["network_architecture"], input_geometry, output_geometry);

  compare_testing.set_classes_count(dataset->get_output_size());
  compare_testing.set_top_n_count(1);
  compare_training.set_classes_count(dataset->get_output_size());
  compare_training.set_top_n_count(1);

  compare_testing_top5.set_classes_count(dataset->get_output_size());
  compare_testing_top5.set_top_n_count(5);
  compare_training_top5.set_classes_count(dataset->get_output_size());
  compare_training_top5.set_top_n_count(5);


  float best_sucess = 0;
  unsigned int epoch_count = parameters.result["epoch_count"].asInt();

  unsigned int sub_epoch_size = 1;
  if (parameters.result["sub_epoch_size"] != Json::Value::null)
    sub_epoch_size = parameters.result["sub_epoch_size"].asInt();

  bool compare_top_5 = false;
  if (parameters.result["compare_top_5"] != Json::Value::null)
    compare_top_5 = parameters.result["compare_top_5"].asBool();

  unsigned int epoch_learning_rate_decay = epoch_count;
  if (parameters.result["epoch_learning_rate_decay"] != Json::Value::null)
    epoch_learning_rate_decay = parameters.result["epoch_learning_rate_decay"].asInt();

  float learning_rate_decay = 1.0;
  if (parameters.result["learning_rate_decay"] != Json::Value::null)
    learning_rate_decay = parameters.result["learning_rate_decay"].asFloat();


  if (parameters.result["preprocessing"] != Json::Value::null)
  {
    t_nn_preprocessing_input.init(input_geometry);
    t_nn_input.init(input_geometry);
    t_nn_required_output.init(output_geometry);

    preprocessing.init(parameters.result["preprocessing"], input_geometry);

    preprocessing_enabled = true;
  }
  else
    preprocessing_enabled = false;





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
  experiment_log << "preprocessing               : " << preprocessing_enabled  << "\n";


  experiment_log << "\n";
  experiment_log << "training\n";


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

        experiment_log << "network success rate top1 " << compare_testing.get_success() << "% " << compare_training.get_success() << "% \n";
        experiment_log << "network success rate top5 " << compare_testing_top5.get_success() << "% " << compare_training_top5.get_success() << "% \n";

        float progress = epoch + sub_epoch*1.0/sub_epoch_size;

        training_progress_log << progress << " " << epoch << " " << sub_epoch << " " << compare_testing.get_success() << " " << compare_training.get_success() << " ";
        training_progress_log << " " << compare_testing_top5.get_success() << " " << compare_training_top5.get_success() << " \n";

        float success = 0.0;
        if (compare_top_5)
          success = compare_testing_top5.get_success();
        else
          success = compare_testing.get_success();

        if (success > best_sucess)
        {
          best_sucess = success;

          if (parameters.result["compare_top_5"].asBool())
            experiment_log << "saving best net in top 5\n";
          else
            experiment_log << "saving best net in top 1\n";

          std::string best_net = config_dir + "trained/";
          nn.save(best_net);

          compare_testing.save_json_file(config_dir + "best_net_testing_result.json");
          compare_testing.save_text_file(config_dir + "best_net_testing_result.txt");

          compare_training.save_json_file(config_dir + "best_net_training_result.json");
          compare_training.save_text_file(config_dir + "best_net_training_result.txt");

          compare_testing_top5.save_json_file(config_dir + "best_net_testing_result_top5.json");
          compare_testing_top5.save_text_file(config_dir + "best_net_testing_result_top5.txt");

          compare_training_top5.save_json_file(config_dir + "best_net_training_result_top5.json");
          compare_training_top5.save_text_file(config_dir + "best_net_training_result_top5.txt");

          experiment_log << "best net saved to " << best_net << ", with success rate " << best_sucess << "%\n";

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


void ClassificationExperiment::train_iterations(CNN &nn, unsigned int iterations)
{
  nn.set_training_mode();

  for (unsigned int i = 0; i < iterations; i++)
  {
    sDatasetItem item = dataset->get_random_training();

    if (preprocessing_enabled)
    {
      t_nn_preprocessing_input.set_from_host(item.input);
      t_nn_required_output.set_from_host(item.output);

      preprocessing.process(t_nn_input, t_nn_preprocessing_input);
      nn.train(t_nn_required_output, t_nn_input);
    }
    else
      nn.train(item.output, item.input);
  }

  nn.unset_training_mode();
}

bool ClassificationExperiment::test(CNN &nn)
{
    compare_testing.clear();
    compare_training.clear();
    compare_testing_top5.clear();
    compare_training_top5.clear();

    bool valid_output = true;

    std::vector<float> nn_output(dataset->get_output_size());

    unsigned int testing_size = dataset->get_testing_size();

    for (unsigned int i = 0; i < testing_size; i++)
    {
      sDatasetItem item = dataset->get_testing(i);

      if (preprocessing_enabled)
      {
        t_nn_preprocessing_input.set_from_host(item.input);
        preprocessing.process(t_nn_input, t_nn_preprocessing_input);

        nn.forward(t_nn_required_output, t_nn_input);

        t_nn_required_output.set_to_host(nn_output);
      }
      else
        nn.forward(nn_output, item.input);


      compare_testing.compare(item.output, nn_output);
      compare_testing_top5.compare(item.output, nn_output);

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
      compare_training_top5.compare(item.output, nn_output);

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


    compare_testing.process(true);
    compare_training.process(true);
    compare_testing_top5.process(true);
    compare_training_top5.process(true);

    return true;
}


void ClassificationExperiment::process_best()
{

}

bool ClassificationExperiment::is_valid(std::vector<float> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    if (isnan(v[i]))
      return false;

  for (unsigned int i = 0; i < v.size(); i++)
    if (isinf(v[i]))
      return false;

  return true;
}
