#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <cnn.h>
#include <timer.h>


void speed_test(unsigned int iterations, bool testing)
{
  sGeometry input_geometry, output_geometry;

  input_geometry.w = 19;
  input_geometry.h = 19;
  input_geometry.d = 4;

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = 19*19+1;

  Tensor t_input(input_geometry);
  Tensor t_output(output_geometry);

  CNN nn("testing_network/parameters.json", input_geometry, output_geometry);

  if (testing)
  {
    nn.set_training_mode();
    for (unsigned int i = 0; i < iterations; i++)
    {
      nn.train(t_output, t_input);
    }
    nn.unset_training_mode();
  }
  else
  {
    for (unsigned int i = 0; i < iterations; i++)
    {
      nn.forward(t_output, t_input);
    }
  }
}

int main()
{
  srand(time(NULL));

  speed_test(10000000, false);

  std::cout << "program done\n";

  return 0;
}
