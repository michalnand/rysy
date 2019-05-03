#include "SignalNet/SignalNet.h"

#include <iostream>
#include <vector>
#include <dataset_line_camera.h>

#include <timer.h>

std::vector<nn_layer_t> float_to_nn_t(std::vector<float> input)
{
  std::vector<nn_layer_t> result(input.size());

  for (unsigned int i = 0; i < input.size(); i++)
  {
    result[i] = input[i]*127;
  }

  return result;
}

unsigned int argmax(std::vector<float> output)
{
  nn_t max = output[0];
  unsigned int result = 0;

  for (unsigned int i = 0; i < output.size(); i++)
    if (output[i] > max)
    {
      max = output[i];
      result = i;
    }

  return result;
}


void speed_test(NeuralNetwork &nn, DatasetInterface &dataset)
{
  unsigned int iterations_count = 1000;
  timer.start();


  for (unsigned int i = 0; i < iterations_count; i++)
  {
    sDatasetItem item = dataset.get_random_testing();
    auto input = float_to_nn_t(item.input);

    nn.set_input(&input[0]);

    nn.forward();
  }

  timer.stop();

  printf("somputing time per iteration %f [ms]\n", timer.get_duration()*1.0/iterations_count);
}

int main()
{
    srand(time(NULL));


    unsigned int sensors_count  = 8;
    unsigned int pixels_count   = 128;
    float noise_level           = 0.4;

    DatasetLineCamera dataset(sensors_count, pixels_count, noise_level);

    dataset.export_h_testing("dataset.h", 100);


    SignalNet nn;

    std::cout << "input size   " << nn.input_size() << "\n";
    std::cout << "output size  " << nn.output_size() << "\n";

    // speed_test(nn, dataset);

    unsigned int good   = 0;
    unsigned int wrong  = 0;
    for (unsigned int i = 0; i < dataset.get_testing_size(); i++)
    {
      sDatasetItem item = dataset.get_testing(i);
      unsigned int dataset_class_result = argmax(item.output);

      auto input = float_to_nn_t(item.input);

      nn.set_input(&input[0]);

      nn.forward();

      if (dataset_class_result == nn.class_result())
        good++;
      else
        wrong++;

      if ((i%100) == 0)
      {
        printf("%6.3f %u %u\n", good*100.0/(good + wrong), good, wrong);
        for (unsigned int j = 0; j < nn.output_size(); j++)
          printf("%6.3f ", nn.get_output()[j]*1.0);
        printf("\n");
      }
    }


  std::cout << "\n";
  std::cout << "GOOD " << good << "\n";
  std::cout << "WRONG " << wrong << "\n";
  std::cout << "result = " << good*100.0/(good + wrong) << "[\%]\n";




  return 0;
}
