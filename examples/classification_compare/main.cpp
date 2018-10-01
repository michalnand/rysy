#include <timer.h>
#include <iostream>

#include <dataset_mnist.h>
#include <dataset_cifar_10.h>

#include <classification_experiment.h>
#include <dataset_pair.h>



int main()
{
  DatasetMnist dataset_source ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                                "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                                "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                                "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                                0);

  unsigned int count = dataset_source.get_training_size()*2;

  std::cout << "count " << count << "\n";
  DatasetPair dataset(dataset_source, count);

  JsonConfig parameters("experiments_mnist_compare.json");

  for (unsigned int i = 0; i < parameters.result["experiments"].size(); i++)
  {
    std::string config_dir = parameters.result["experiments"][i].asString();
    ClassificationExperiment experiment(dataset, config_dir);
    experiment.run();
  }

  std::cout << "program done\n";

  return 0;
}
