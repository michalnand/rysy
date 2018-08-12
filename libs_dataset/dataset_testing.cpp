#include "dataset_testing.h"

#include <math.h>

DatasetTesting::DatasetTesting(unsigned int dim, float output_amplitude)
               :DatasetInterface()
{
  channels  = 1;
  width     = dim;
  height    = 1;
  this->output_amplitude = output_amplitude;

  amplitude.resize(get_input_size());
  frequency.resize(get_input_size());
  phase.resize(get_input_size());

  for (unsigned int i = 0; i < get_input_size(); i++)
  {
    amplitude[i]  = 1.0 + 0.2*rnd();
    frequency[i]  = 10.0*rnd();
    phase[i]      = 2.0*3.141592654*rnd();
  }

  unsigned int items_count = 10000;

  for (unsigned int i = 0; i < items_count; i++)
    training.push_back(create_item());

  for (unsigned int i = 0; i < items_count; i++)
    testing.push_back(create_item());

  for (unsigned int i = 0; i < 20; i++)
    print_training_item(i%training.size());

}


DatasetTesting::~DatasetTesting()
{

}


sDatasetItem DatasetTesting::create_item()
{
  struct sDatasetItem result;

  result.input.resize(get_input_size());
  result.output.resize(1);

  for (unsigned int i = 0; i < result.input.size(); i++)
    result.input[i] = rnd();


  float y = 0.0;
  for (unsigned int i = 0; i < result.input.size(); i++)
    y+= amplitude[i]*sin(frequency[i]*result.input[i] + phase[i]);

  result.output[0] = (y/result.input.size())*output_amplitude;

  return result;
}

float DatasetTesting::rnd()
{
  return ((rand()%2000000) - 1000000.0)/1000000.0;
}
