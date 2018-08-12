#include <stdio.h>

#include <dataset_mnist.h>
#include <dataset_stl10.h>
#include <autoencoder_train.h>

int main()
{
  srand(time(NULL));


  printf("starting\n");

//  DatasetSTL10   dataset(8);


  DatasetMnist             dataset( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                                    "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                                    "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                                    "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                                   2);

  AutoencoderTrain autoencoder_train(&dataset, "mnist_autoencoder/");

   autoencoder_train.main();


  printf("program done\n");

  return 0;
}
