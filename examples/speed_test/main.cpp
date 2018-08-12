#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <iostream>

#include <dataset_mnist.h>
#include <dataset_cifar_10.h>
#include <cnn.h>


int main()
{
  srand(time(NULL));


  DatasetMnist dataset ( "/home/michal/dataset/mnist/train-images.idx3-ubyte",
                         "/home/michal/dataset/mnist/train-labels.idx1-ubyte",
                         "/home/michal/dataset/mnist/t10k-images.idx3-ubyte",
                         "/home/michal/dataset/mnist/t10k-labels.idx1-ubyte",
                         0);

  sGeometry input_geometry, output_geometry;

  input_geometry.w = dataset.get_width();
  input_geometry.h = dataset.get_height();
  input_geometry.d = dataset.get_channels();

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset.get_output_size();

  CNN nn("network_architecture.json", input_geometry, output_geometry);

  nn.set_training_mode();
  for (unsigned int i = 0; i < 100; i++)
  {
    sDatasetItem item = dataset.get_random_training();
    nn.train(item.output, item.input);
  }
  nn.unset_training_mode();

  printf("program done\n");

  return 0;
}
