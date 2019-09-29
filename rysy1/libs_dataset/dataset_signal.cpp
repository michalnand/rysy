#include "dataset_signal.h"
#include <math.h>

DatasetSignal::DatasetSignal(unsigned int classes_count, unsigned int length, unsigned int channels, float noise_level)
{
  this->width      = length;
  this->height     = 1;
  this->channels   = channels;

  output_size = classes_count;
  training.resize(output_size);

  unsigned int total_count = classes_count*10000;

  create(total_count, length, 0.1, noise_level);

  print();


  for (unsigned int i = 0; i < 10; i++)
    print_testing_item(i);

}


DatasetSignal::~DatasetSignal()
{

}

void DatasetSignal::create(unsigned int items_count, unsigned int length, float testing_ratio, float noise_level)
{
    sDatasetItem item;

    item.output.resize(output_size);

    for (unsigned int i = 0; i < items_count; i++)
    {
        for (unsigned int j = 0; j < item.output.size(); j++)
            item.output[j] = 0.0;

        unsigned int class_id = rand()%output_size;
        float frequency = class_id + 4;

        item.input = signal(frequency, length, channels, noise_level);
        item.output[class_id] = 1.0;

        float p = (rand()%100000)/100000.0;
        if (p < testing_ratio)
            add_testing(item);
        else
            add_training(item);
    }
}


std::vector<float> DatasetSignal::signal(float frequency, unsigned int length, unsigned int channels, float noise_level)
{
    std::vector<float> result(length*channels);

    float pi = 3.141592654;

    for (unsigned int ch = 0; ch < channels; ch++)
    {
      float phase = ((rand()%100000)/100000.0)*2.0*pi;

      for (unsigned int i = 0; i < length; i++)
      {
          float phase_noise = 0.1*((rand()%100000)/100000.0 - 0.5)*2.0;

          float x = phase_noise + phase + (2.0*pi*frequency*i)/length;
          float y = sin(x);

          float noise = ((rand()%100000)/100000.0 - 0.5)*2.0;

          result[i + ch*length] = (1.0 - noise_level)*y + noise_level*noise;
      }
    }

    float max = result[0];
    float min = result[0];
    for (unsigned int i = 0; i < result.size(); i++)
    {
        if (result[i] > max)
            max = result[i];

        if (result[i] < min)
            min = result[i];
    }

    float k = 0.0;
    float q = 0.0;

    if (max > min)
    {
        k = 1.0/(max - min);
        q = 1.0 - k*max;
    }

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = k*result[i] + q;

    return result;
}
