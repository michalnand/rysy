#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <iostream>

#include <log.h>
#include <layers/convolution_layer.h>

Log result_log("result.log");

double layer_test(sGeometry input_geometry, sGeometry kernel_geometry)
{
  double result;

  sHyperparameters hyperparameters;

  hyperparameters.init_weight_range = 0.0;
  hyperparameters.learning_rate = 0.001;
  hyperparameters.lambda1 = 0.0001;
  hyperparameters.lambda2 = 0.0001;
  hyperparameters.dropout = 0.0;

  hyperparameters.beta1   = 0.9;
  hyperparameters.beta2   = 0.999;
  hyperparameters.epsilon = 0.00000001;

  hyperparameters.minibatch_size  = 32;

  result_log << input_geometry.w << " ";
  result_log << input_geometry.d << " ";
  result_log << kernel_geometry.w << " ";
  result_log << kernel_geometry.d << " ";

  ConvolutionLayer layer(input_geometry, kernel_geometry, hyperparameters);

  Tensor input(layer.get_input_geometry());
  Tensor output(layer.get_output_geometry());

  unsigned int iterations = 1000;
  timer.start();

  for (unsigned int i = 0; i < iterations; i++)
    layer.forward(output, input);

  timer.stop();

  result = timer.get_duration()*1.0/iterations;

  result_log << result << " ";

  result_log << "\n";

  return result;
}

int main()
{
  srand(time(NULL));

  sGeometry input_geometry;
  sGeometry kernel_geometry;

  double average_time = 0.0;
  double max_time = 0.0;
  double min_time = 1000000000.0;

  double count = 0.0;

  unsigned int input_feature_maps_count = 32;
  unsigned int kernel_count = 32;

  for (unsigned int kernel_sizes = 0; kernel_sizes < 3; kernel_sizes++)
//  for (unsigned int input_feature_maps_count = 1; input_feature_maps_count < 32; input_feature_maps_count*= 2)
    for (unsigned int input_size = 8; input_size <= 64; input_size*= 2)
//      for (unsigned int kernel_count = 8; kernel_count < 48; kernel_count+= 8)
      { 
        unsigned int kernel_size = 1;

        switch (kernel_sizes)
        {
          case 0: kernel_size = 1; break;
          case 1: kernel_size = 3; break;
          case 2: kernel_size = 5; break;
        }

        kernel_geometry.w = kernel_size;
        kernel_geometry.h = kernel_size;
        kernel_geometry.d = kernel_count;

        input_geometry.w = input_size;
        input_geometry.h = input_size;
        input_geometry.d = input_feature_maps_count;

        float time_ = layer_test(input_geometry, kernel_geometry);

        average_time+= time_;

        if (time_ < min_time)
          min_time = time_;

        if (time_ > max_time)
          max_time = time_;

        count++;
      }

  average_time/= count;

  result_log << "\n";
  result_log << "average time " << average_time << "\n";
  result_log << "min time " << min_time << "\n";
  result_log << "max time " << max_time << "\n";

  return 0;
}
