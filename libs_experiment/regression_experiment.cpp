#include "regression_experiment.h"

#include <timer.h>
#include <math.h>

RegressionExperiment::RegressionExperiment(std::string parameters_file_name, DatasetInterface *dataset)
{
  JsonConfig json(parameters_file_name);
  this->dataset = dataset;

  parameters = json.result;

  init();
}


void RegressionExperiment::init()
{
  log_prefix = parameters["experiment_log_prefix"].asString();

  experiment_log.set_output_file_name(log_prefix + "experiment.log");
  training_log.set_output_file_name(log_prefix + "training_progress.log");

  experiment_log << "initializing\n";

  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset->get_output_size();


  unsigned int batch_max_size  = parameters["batch_max_size"].asInt();

  batch = new Batch(input_geometry,
                    output_geometry,
                    batch_max_size);


  experiment_log << "creating batch with size " << batch->size() << "\n";

  fill_batch();


  init_networks_try_count   = parameters["init_networks_try_count"].asInt();
  epoch_count               = parameters["epoch_count"].asInt();
  learning_rate_decay       = parameters["learning_rate_decay"].asFloat();

  learning_rate       = parameters["network_architecture"]["hyperparameters"]["learning_rate"].asFloat();
  lambda1              = parameters["network_architecture"]["hyperparameters"]["lambda1"].asFloat();
  lambda2              = parameters["network_architecture"]["hyperparameters"]["lambda2"].asFloat();

  experiment_log << "\n";
  experiment_log << "parameters :\n";

  experiment_log << "init networks try count : " << init_networks_try_count << "\n";
  experiment_log << "epoch count :             " << epoch_count << "\n";

  experiment_log << "learning rate_decay : " << learning_rate_decay << "\n";

  experiment_log << "init done\n\n";
}

RegressionExperiment::~RegressionExperiment()
{
  experiment_log << "uninit\n";

  if (batch != nullptr)
  {
    delete batch;
    batch = nullptr;
  }

  experiment_log << "uninit done\n";
}



void RegressionExperiment::run()
{
  srand(time(NULL));

  error_best = 1000000000.0;

  //skip search init network when there is pretrained net
  if (parameters["use_pretrained"].asBool() == false)
  {
    if (search_initial_net() != 0)
      return;
  }


  std::string best_net_file_name;
  best_net_file_name = log_prefix + "trained/";

  experiment_log << "loading from : " << best_net_file_name << "cnn_config.json" << "\n";
  CNN *nn = new CNN(best_net_file_name + "cnn_config.json", input_geometry, output_geometry);

  /*

  TODO : set learning_rate
  learning_rate = nn->get_learning_rate();
  lambda        = nn->get_lambda();
  */
  epoch_without_improvement = 0;

  experiment_log << "training\n\n";


  //learn network -> use best network
  for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
  {
    experiment_log << "epoch " << epoch << "\n";

    fill_batch();

    nn->set_training_mode();
    //learning
    for (unsigned int i = 0; i < batch->size(); i++)
    {
      batch->set_random();
      nn->train(batch->get_output(), batch->get_input());

      if ((i%(batch->size()/100)) == 0)
      {
        sRegressionTestResult quick_result = test(nn, true);

        float done = (100.0*i)/batch->size();
        experiment_log << "training done = " << done << " %    error = " << quick_result.error << " \n";
      }
    }

    nn->unset_training_mode();

    //process testing on testing set
    experiment_log << "testing\n";
    result = test(nn);

    //NaN error -> set lower learning rate and load best network
    if (result.output_valid == false)
    {
      delete nn;
      nn = load_saved_net(best_net_file_name);
    }
    //error too big -> set lower learning rate and load best network
    else if (result.too_big_difference)
    {
      delete nn;
      nn = load_saved_net(best_net_file_name);
    }
    //best network -> save result
    else if (result.best_net)
    {
      nn->save(best_net_file_name);

      experiment_log << " error = " << result.error << " saved as new best network\n";

      epoch_without_improvement = 0;

      best_result = result;

      best_net_log_process();

      save_examples(nn);
    }
    else if (result.long_term_without_improvement)
    {
      delete nn;
      nn = load_saved_net(best_net_file_name);
    }

    experiment_log << " error = " << result.error << " " << result.error_min << " " << result.error_max << "\n";
    training_log << epoch << " " << result.error << " " << result.error_min << " " << result.error_max << "\n";
    experiment_log << "\n";
  }

  delete nn;
  experiment_log << "training done\n";
}




int RegressionExperiment::search_initial_net()
{
  bool saved = false;

  std::string best_net_file_name;
  best_net_file_name = log_prefix + "trained/";

  experiment_log << "search initial network\n\n";

  //try learn few networks one epoch
  for (unsigned int nn_try = 0; nn_try < init_networks_try_count; nn_try++)
  {
    experiment_log << "creating new net with id " << nn_try << "\n";

    CNN *nn = new CNN(parameters["network_architecture"], input_geometry, output_geometry);

    experiment_log << "training network\n";

    //learning
    for (unsigned int i = 0; i < batch->size(); i++)
    {
      batch->set_random();
      nn->train(batch->get_output(), batch->get_input());

      if ((i%(batch->size()/100)) == 0)
      {
        sRegressionTestResult quick_result = test(nn, true);

        float done = (100.0*i)/batch->size();
        experiment_log << "training done = " << done << " %    error = " << quick_result.error << " \n";
      }
    }

    experiment_log << "testing network\n";

    //process testing on testing set
    sRegressionTestResult compare_result = test(nn);

    if (compare_result.output_valid)
    {
      if (compare_result.best_net)
      {
        nn->save(best_net_file_name);

        best_result = result;
        best_net_log_process();

        save_examples(nn);

        saved = true;
        experiment_log << "net " << nn_try << " error = " << compare_result.error << " saved as new best network\n";
      }
      else
        experiment_log << "net " << nn_try << " error = " << compare_result.error << "\n";
    }
    else
    {
      experiment_log << "net " << nn_try << " " << "NaN\n";
    }

    delete nn;

    experiment_log << "\n\n\n";
  }

  //all nets ended as NaN -> terminate experiment, try lower learning rate or initial weights range
  if (saved == false)
  {
    experiment_log << "ending experiment -> no valid network\n";
    return -1;
  }

  return 0;
}

CNN* RegressionExperiment::load_saved_net(std::string best_net_file_name)
{
  learning_rate*= learning_rate_decay;
  lambda1*= learning_rate_decay;
  lambda2*= learning_rate_decay;

  if (result.output_valid != true)
    experiment_log << "NaN error, loading best network, and setting learning rate = " << learning_rate << "\n";
  else if (result.too_big_difference)
    experiment_log << "too big error " << result.error <<" , loading best network and setting learning rate = " << learning_rate << "\n";
  else if (result.long_term_without_improvement)
    experiment_log << "long term without improvement, loading best network, and setting learning rate = " << learning_rate << "\n";
  else
    experiment_log << "something wrong\n";

  CNN *nn = new CNN(best_net_file_name + "cnn_config.json", input_geometry, output_geometry);
  nn->set_learning_rate(learning_rate);
  nn->set_lambda1(lambda1);
  nn->set_lambda2(lambda2);

  epoch_without_improvement = 0;

  return nn;
}


sRegressionTestResult RegressionExperiment::test(CNN *nn, bool quick_test)
{
  sRegressionTestResult result;

  result.output_valid       = true;
  result.too_big_difference = false;
  result.best_net           = false;
  result.error              = 10000000.0;
  result.error_min          = result.error;
  result.error_max          = -result.error;

  result.long_term_without_improvement = false;


  std::vector<float> nn_output(dataset->get_output_size());

  float error_sum = 0.0;

  unsigned int testing_iterations = dataset->get_testing_size();
  if (quick_test)
    testing_iterations/= 10;

  for (unsigned int i = 0; i < testing_iterations; i++)
  {
    sDatasetItem ground_truth;

    if (quick_test)
      ground_truth = dataset->get_random_testing();
    else
      ground_truth = dataset->get_testing(i);

    nn->forward(nn_output, ground_truth.input);

    if (check_valid(nn_output) != true)
    {
      result.output_valid = false;
      break;
    }

    float error = rms(ground_truth.output, nn_output);

    if (error > result.error_max)
      result.error_max = error;

    if (error < result.error_min)
      result.error_min = error;

    error_sum+= error;

/*
    if ((i%10) == 0)
    {
      print_vector(ground_truth.output);
      printf(" : ");
      print_vector(nn_output);
      printf("\n");
    }
*/
  }


  result.error = error_sum/testing_iterations;

  if (quick_test)
    return result;

  if (result.error < error_best)
  {
    result.best_net = true;
    error_best = result.error;
  }
  else
  if (result.error > error_best*1.2)
  {
    result.too_big_difference = true;
  }
  else
  {
    epoch_without_improvement++;
    if ((int)epoch_without_improvement >= parameters["epoch_without_improvement"].asInt())
    {
      epoch_without_improvement = 0.0;
      result.long_term_without_improvement = true;
    }
  }

  return result;
}

bool RegressionExperiment::check_valid(std::vector<float> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    if (isnan(v[i]))
      return false;

  for (unsigned int i = 0; i < v.size(); i++)
    if (isinf(v[i]))
      return false;

  return true;
}

void RegressionExperiment::fill_batch()
{
  for (unsigned int i = 0; i < batch->size(); i++)
  {
    sDatasetItem item = dataset->get_random_training();
    batch->add(item.output, item.input);
  }
}


float RegressionExperiment::rms(std::vector<float> &va, std::vector<float> &vb)
{
  float result = 0.0;

  for (unsigned int i = 0; i < va.size(); i++)
  {
    float tmp = va[i] - vb[i];
    result+= tmp*tmp;
  }

  return sqrt(result/va.size());
}


void RegressionExperiment::print_vector(std::vector<float> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    printf("%6.3f ", v[i]);
}

void RegressionExperiment::best_net_log_process()
{

}

void RegressionExperiment::save_examples(CNN *nn)
{
  Log output_examples("best_net_output_examples.log");


  std::vector<float> nn_output(dataset->get_output_size());

  for (unsigned int i = 0; i < 100; i++)
  {
    sDatasetItem ground_truth;
    ground_truth = dataset->get_random_testing();
    nn->forward(nn_output, ground_truth.input);

//    output_examples << "\n\n";
//    output_examples << ground_truth.input << "\n";
    output_examples << ground_truth.output << " >> " << nn_output << "\n";
  }



}
