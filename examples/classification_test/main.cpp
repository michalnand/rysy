#include <timer.h>
#include <iostream>

#include <dataset_mnist.h>
#include <dataset_cifar_10.h>

#include <classification_experiment.h>

#include <dataset_line.h>

int main()
{
  DatasetLine dataset;

  {
    ClassificationExperiment experiment(dataset, "line_type_network/");
    experiment.run();
  }




  /*
  {
    DatasetMnist dataset ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                           "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                          0);

    JsonConfig parameters("experiments_mnist.json");

    for (unsigned int i = 0; i < parameters.result["experiments"].size(); i++)
    {
      std::string config_dir = parameters.result["experiments"][i].asString();
      ClassificationExperiment experiment(dataset, config_dir);
      experiment.run();
    }
  }
  */

  std::cout << "program done\n";

  return 0;
}
