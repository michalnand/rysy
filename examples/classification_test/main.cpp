#include <iostream>
#include <dataset_mnist.h>
#include <classification_experiment.h>


void net_test(DatasetInterface &dataset, std::string config_dir)
{
  JsonConfig parameters(config_dir+"parameters.json");

  sGeometry input_geometry, output_geometry;

  input_geometry.w = dataset.get_width();
  input_geometry.h = dataset.get_height();
  input_geometry.d = dataset.get_channels();

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset.get_output_size();

  CNN nn(parameters.result["network_architecture"], input_geometry, output_geometry);

  unsigned int iterations = 1000;

  nn.set_training_mode();

  for (unsigned int i = 0; i < iterations; i++)
  {
    sDatasetItem item = dataset.get_random_training();

    nn.train(item.output, item.input);
  }

  nn.unset_training_mode();
}

int main()
{
    DatasetMnist dataset ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                           "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                           true);


    JsonConfig parameters("experiments_mnist.json");

    for (unsigned int i = 0; i < parameters.result["experiments"].size(); i++)
    {
      std::string config_dir = parameters.result["experiments"][i].asString();
      ClassificationExperiment experiment(dataset, config_dir);
      experiment.run();
    }



  std::cout << "program done\n";

  return 0;
}
