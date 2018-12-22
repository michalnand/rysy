#include <iostream>
#include <dataset_cifar.h>
#include <classification_experiment.h>


int main()
{
    DatasetCIFAR dataset ( "/home/michal/dataset/cifar/train.bin",
                           "/home/michal/dataset/cifar/test.bin",
                           true
                         );
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

    {
      ClassificationExperiment experiment(dataset, "cifar_4/");
      experiment.run();
    }



  std::cout << "program done\n";

  return 0;
}
