#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
#include <iostream>

#include <log.h>
#include <layers/convolution_layer.h>


void layer_forward_test(sGeometry input_geometry, sGeometry kernel_geometry)
{
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

  ConvolutionLayer layer(input_geometry, kernel_geometry, hyperparameters);

  layer.w.set_const(1.0);
  layer.bias.set_const(0.0);

  Tensor input(layer.get_input_geometry());
  Tensor output(layer.get_output_geometry());

  input.set_const(1.0);
  //input.set_random(1.0);

  input.print();

  layer.forward(output, input);

  output.print();
}




int main()
{
  srand(time(NULL));

  sGeometry input_geometry, kernel_geometry;

  kernel_geometry.w = 3;
  kernel_geometry.h = 3;
  kernel_geometry.d = 2;

  {
      input_geometry.w = 9;
      input_geometry.h = 9;
      input_geometry.d = 1;


      layer_forward_test(input_geometry, kernel_geometry);
    }

    {
        input_geometry.w = 10;
        input_geometry.h = 10;
        input_geometry.d = 1;

        layer_forward_test(input_geometry, kernel_geometry);
      }

      {
          input_geometry.w = 19;
          input_geometry.h = 19;
          input_geometry.d = 1;

          layer_forward_test(input_geometry, kernel_geometry);
        }

        {
            input_geometry.w = 20;
            input_geometry.h = 20;
            input_geometry.d = 1;

            layer_forward_test(input_geometry, kernel_geometry);
          }
  return 0;
}
