#include <iostream>
#include <dataset_mnist.h>
#include <regression_experiment.h>


int main()
{
    DatasetMnist dataset ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                           "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                           "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                          0);


  RegressionExperiment experiment(dataset, "mnist_0/");
  experiment.run();

  std::cout << "program done\n";

  return 0;
}
