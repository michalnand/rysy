#include <iostream>
#include <dataset_preprocessing.h>
#include <dataset_cifar.h>
#include <classification_experiment.h>


int main()
{
    DatasetCIFAR dataset_raw ( "/home/michal/dataset/cifar/train.bin",
                           "/home/michal/dataset/cifar/test.bin",
                           true
                         );

    DatasetPreprocessing dataset (dataset_raw, "preprocessing.json");

   //dataset.save_images("/home/michal/cifar/training/", "/home/michal/cifar/testing/");


    {
      ClassificationExperiment experiment(dataset, "cifar_0/");
      experiment.run();
    }

    {
      ClassificationExperiment experiment(dataset, "cifar_1/");
      experiment.run();
    }

    {
      ClassificationExperiment experiment(dataset, "cifar_2/");
      experiment.run();
    }

    {
      ClassificationExperiment experiment(dataset, "cifar_3/");
      experiment.run();
    }





  std::cout << "program done\n";

  return 0;
}
