#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <iostream>

#include <dataset_mnist.h>
#include <dataset_cifar_10.h>
#include <dataset_images.h>

#include <classification_experiment.h>


int main()
{
  srand(time(NULL));

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


  DatasetMnist dataset ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                         "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                         "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                         "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                        0);



  ClassificationExperiment experiment(dataset, "kernel_test/");
  experiment.run();


  printf("program done\n");

  return 0;
}
