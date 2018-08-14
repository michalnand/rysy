#include <stdio.h>
#include <dataset_mnist.h>
#include <dataset_images.h>
#include <autoencoder_experiment.h>

int main()
{
  srand(time(NULL));


  /*
  DatasetMnist dataset( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                        "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                        "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                        "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                       0);

  AutoencoderExperiment autoencoder(dataset, "mnist_0/");

  autoencoder.run();
  */

  DatasetImages dataset("holes_0/dataset_parameters.json");

  AutoencoderExperiment autoencoder(dataset, "holes_0/");

  autoencoder.run();

  printf("program done\n");

  return 0;
}
