#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <iostream>

#include <dataset_mnist.h>
#include <dataset_cifar_10.h>
#include <classification_experiment.h>


int main()
{
  srand(time(NULL));
  /*
  {
    DatasetCIFAR10 dataset ( "/home/michal/dataset/cifar_10/data_batch_1.bin",
                             "/home/michal/dataset/cifar_10/data_batch_2.bin",
                             "/home/michal/dataset/cifar_10/data_batch_3.bin",
                             "/home/michal/dataset/cifar_10/data_batch_4.bin",
                             "/home/michal/dataset/cifar_10/data_batch_5.bin",

                             "/home/michal/dataset/cifar_10/test_batch.bin",
                             0);

   printf("dataset loading done\n");
    JsonConfig parameters("experiments_cifar.json");

    for (unsigned int i = 0; i < parameters.result["experiments"].size(); i++)
    {
      std::string config_dir = parameters.result["experiments"][i].asString();
      ClassificationExperiment experiment(dataset, config_dir);
      experiment.run();
    }
  }
  */


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

  printf("program done\n");

  return 0;
}
