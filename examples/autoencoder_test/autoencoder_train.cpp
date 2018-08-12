#include "autoencoder_train.h"
#include <string>
#include <vector>
#include <stdio.h>

#include <image.h>
#include <layers/convolution_layer.h>
#include <timer.h>

AutoencoderTrain::AutoencoderTrain(DatasetInterface *dataset, std::string experiment_path)
{
  //experiment_log.set_output_file_name("experiment.log");
  this->experiment_path = experiment_path;

  experiment_log.set_output_file_name(experiment_path + "experiment.log");

  experiment_log << "init\n";

  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  output_geometry.w = dataset->get_width();
  output_geometry.h = dataset->get_height();
  output_geometry.d = dataset->get_channels();

  sGeometry batch_output_geometry;
  batch_output_geometry.w = 1;
  batch_output_geometry.h = 1;
  batch_output_geometry.d = dataset->get_output_size();

  experiment_log << "creating training batch\n";

  training_batch = new Batch(input_geometry, batch_output_geometry, dataset->get_unlabeled_size(), 0.0);
  for (unsigned int i = 0; i < dataset->get_unlabeled_size(); i++)
  {
    sDatasetItem item = dataset->get_unlabeled(i);
    training_batch->add(item.output, item.input);

    if ((i%1000) == 0)
    {
      float done = i*100.0/dataset->get_unlabeled_size();
      experiment_log << "creating training batch " << done << "% \n";
    }

  }

  experiment_log << "training batch with " << training_batch->size() << " items \n";
  experiment_log << "\n";

  experiment_log << "creating testing batch\n";

  testing_batch = new Batch(input_geometry, batch_output_geometry, dataset->get_testing_size(), 0.0);

  for (unsigned int i = 0; i < dataset->get_testing_size(); i++)
  {
    sDatasetItem item = dataset->get_testing(i);
    testing_batch->add(item.output, item.input);
  }

  experiment_log << "testing batch with " << testing_batch->size() << " items \n";
  experiment_log << "\n";

  experiment_log << "saving dataset examples\n";

  for (unsigned int i = 0; i < 32; i++)
  {
    std::string file_name;
    file_name = experiment_path + "dataset_examples/" + std::to_string(i) + ".jpg";
    training_batch->set(i);
    save_image(file_name, training_batch->get_input());
  }
  experiment_log << "\n";

  experiment_log << "init done\n";
}

AutoencoderTrain::~AutoencoderTrain()
{
  delete training_batch;
  delete testing_batch;

  experiment_log << "uninit done\n";
}

void AutoencoderTrain::main()
{
  std::string network_config_file_name;
  network_config_file_name = experiment_path + "cnn_config.json";

  JsonConfig config_json(network_config_file_name);

  unsigned int epoch_count = config_json.result["epoch_count"].asInt();
  unsigned int learning_iterations = training_batch->size();

  unsigned int input_size = input_geometry.w*input_geometry.h*input_geometry.d;

  rms       = input_size*1000.0;
  rms_max   = input_size*1000.0;
  rms_best  = input_size*1000.0;
  rms_min   = input_size*1000.0;

  experiment_log << "\n";
  experiment_log << "search initial networks\n";

  unsigned int net_try_max  = config_json.result["init_networks_try_count"].asInt();

  for (unsigned int net_try = 0; net_try < net_try_max; net_try++)
  {
    CNN nn(config_json.result["network_architecture"], input_geometry, output_geometry);


    for (unsigned int i = 0; i < learning_iterations; i++)
    {
      training_batch->set_random();
      nn.train(training_batch->get_input(), training_batch->get_input());
    }

    test(nn, true);

    experiment_log << "init net id " << net_try << " rms " << rms << "\n";

    if (rms < rms_best)
    {
      rms_best = rms;
      nn.save(experiment_path + "trained/");

      experiment_log << "new best init net id " << net_try << "\n";

      test_images(experiment_path + "testing_examples/");
    }
  }

  experiment_log << "\n";

  experiment_log << "loading net\n";


  CNN *nn = new CNN(experiment_path + "trained/cnn_config.json", input_geometry, output_geometry);


  experiment_log << "training start\n";

  for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
  {
    experiment_log << "training epoch " << epoch << "\n";

    for (unsigned int i = 0; i < learning_iterations; i++)
    {
      training_batch->set_random();
      nn->train(training_batch->get_input(), training_batch->get_input());
    }

    //quick check for rms, rms_max
    test(*nn, true);

    std::string log_msg;

    if ((nan_error)||(rms > 1.2*rms_best))
    {
      delete nn;

      nn = new CNN(experiment_path + "trained/cnn_config.json", input_geometry, output_geometry);

      log_msg = "nan error, loaded best saved network\n";
    }
    else
    if (rms < rms_best)
    {
      rms_best = rms;
      nn->save(experiment_path + "trained/");
      log_msg = "new best net saved";

      test_images(experiment_path + "testing_examples/");
    }

    experiment_log << epoch << " "
                   << epoch*100.0/epoch_count << " "
                   << rms     << " "
                   << rms_min << " "
                   << rms_max << " "
                   << rms_best << " "
                   << log_msg << "\n";

  }

  test(*nn, false);


  experiment_log << "final test "
                 << rms     << " "
                 << rms_min << " "
                 << rms_max << " "
                 << rms_best << "\n";

  delete nn;


  experiment_log << "training done\n";

  test_images(experiment_path + "testing_examples/");

  experiment_log << "testing samples done\n";
  experiment_log << "done\n";
}


void AutoencoderTrain::test(CNN &nn, bool quick)
{
  unsigned int ratio = 1;
  if (quick)
    ratio = 5;

  unsigned int testing_iterations = testing_batch->size()/ratio;

  Tensor t_nn_output(nn.get_output_geometry());

  rms     = 0.0;
  rms_max = 0.0;
  rms_min = 1000000000.0;

  nan_error = false;

  for (unsigned int i = 0; i < testing_iterations; i++)
  {
    if (quick)
      testing_batch->set_random();
    else
      testing_batch->set(i);


    nn.forward(t_nn_output, testing_batch->get_input());

    if ((i%100) == 0)
    if (t_nn_output.check_valid() != true)
      nan_error = true;

    float tmp = t_nn_output.get_rms(testing_batch->get_input());

    rms+= tmp;

    if (tmp > rms_max)
      rms_max = tmp;

    if (tmp < rms_min)
      rms_min = tmp;
  }

  rms = rms/testing_iterations;
}



void AutoencoderTrain::test_images(std::string result_path)
{
  CNN nn(experiment_path + "trained/cnn_config.json", input_geometry, output_geometry);

  unsigned int testing_iterations = 32;

  Tensor t_nn_output(nn.get_output_geometry());

  for (unsigned int i = 0; i < testing_iterations; i++)
  {
    unsigned int item_idx = i;

    testing_batch->set(item_idx);

    nn.forward(t_nn_output, testing_batch->get_input());

    std::string required_output_image_file_name;
    std::string output_image_file_name;
    required_output_image_file_name  = result_path + std::to_string(item_idx) + "_required.jpg";
    output_image_file_name = result_path + std::to_string(item_idx) + "_out.jpg";

    save_image(required_output_image_file_name, testing_batch->get_input());
    save_image(output_image_file_name, t_nn_output);
  }
}


void AutoencoderTrain::export_all()
{
  export_batch("/home/michal/dataset/mnist2/training/", training_batch);
  export_batch("/home/michal/dataset/mnist2/testing/" , testing_batch);
}

void AutoencoderTrain::export_batch(std::string export_path, Batch *batch)
{
  CNN nn(experiment_path + "trained/cnn_config.json", input_geometry, output_geometry);

  Tensor t_nn_output(nn.get_output_geometry());

  for (unsigned int item_idx = 0; item_idx < batch->size(); item_idx++)
  {
    batch->set(item_idx);

    unsigned int output_num = output_transform(batch->get_output());


    nn.forward(t_nn_output, batch->get_input());

    std::string required_output_image_file_name;
    std::string output_image_file_name;


    required_output_image_file_name  = export_path + std::to_string(output_num) + "_" + std::to_string(item_idx) + "_required.png";
    output_image_file_name           = export_path + std::to_string(output_num) + "_" + std::to_string(item_idx) + "_nnout.png";

    save_image(required_output_image_file_name, batch->get_input());
    save_image(output_image_file_name, t_nn_output);

    if ((item_idx%100) == 0)
      printf("export to %s %s %6.3f %%\n", required_output_image_file_name.c_str(), output_image_file_name.c_str(), item_idx*100.0/batch->size());
  }
}

unsigned int AutoencoderTrain::output_transform(Tensor &tensor)
{
  std::vector<float> tmp;
  tmp.resize(tensor.size);
  tensor.set_to_host(tmp);

  unsigned int result = 0;
  for (unsigned int i = 0; i < tmp.size(); i++)
  {
    if (tmp[i] > 0.0)
    {
      result = i;
      break;
    }
  }
  return result;
}



void AutoencoderTrain::test_image(std::string input_file_name, std::string output_file_name)
{
  Image input_image(input_file_name);

  sGeometry input_geometry;
  input_geometry.w = input_image.width;
  input_geometry.h = input_image.height;
  input_geometry.d = 3;


  Tensor input(input_geometry);
  Tensor output(input_geometry);

  std::vector<float> image_v = input_image.as_vector();
  input.set_from_host(image_v);

  CNN nn(experiment_path + "trained/cnn_config.json", input_geometry, input_geometry);

  timer.start();
  nn.forward(output, input);
  timer.stop();

  experiment_log << "test image forward time " << timer.get_duration() << " [ms]\n";

  save_image(output_file_name, output);
}

void AutoencoderTrain::save_image(std::string file_name, Tensor &tensor)
{
  std::vector<float> v_result;
  v_result.resize(tensor.size);

  tensor.set_to_host(v_result);

  Image image(tensor.w, tensor.h);

  if (tensor.d == 1)
    image.from_vector_grayscale(v_result);
  else
    image.from_vector(v_result);

  image.save(file_name);
}
