#include "dataset_line_camera.h"
#include <math.h>

DatasetLineCamera::DatasetLineCamera(unsigned int sensors_count, unsigned int pixels_count, float noise_level)
{
  width      = pixels_count;
  height     = 1;
  channels   = 1;

  output_size = sensors_count;
  training.resize(output_size);

  unsigned int total_count = sensors_count*10000;

  create(total_count, pixels_count, 0.1, noise_level);

  print();


  for (unsigned int i = 0; i < 10; i++)
    print_testing_item(i);

}


DatasetLineCamera::~DatasetLineCamera()
{

}

void DatasetLineCamera::create(unsigned int items_count, unsigned int pixels_count, float testing_ratio, float noise_level)
{
    sDatasetItem item;

    item.output.resize(output_size);

    for (unsigned int i = 0; i < items_count; i++)
    {
        for (unsigned int j = 0; j < item.output.size(); j++)
            item.output[j] = 0.0;

        unsigned int class_id = rand()%output_size;
        float center = ((class_id*1.0/output_size) - 0.5)*2.0;

        item.input = signal(center, pixels_count, noise_level);
        item.output[class_id] = 1.0;

        float p = (rand()%100000)/100000.0;
        if (p < testing_ratio)
            add_testing(item);
        else
            add_training(item);
    }
}


std::vector<float> DatasetLineCamera::signal(float center, unsigned int length, float noise_level)
{
    std::vector<float> result(length);

    for (unsigned int i = 0; i < length; i++)
    {
        float k = 40.0;
        float x = ((i*1.0/length) - 0.5)*2.0;
        float d = x - center;
        float y = exp(-k*d*d);

        float noise = ((rand()%100000)/100000.0 - 0.5)*2.0;

        result[i] = (1.0 - noise_level)*y + noise_level*noise;
    }

    float max = result[0];
    float min = result[0];
    for (unsigned int i = 0; i < length; i++)
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

    for (unsigned int i = 0; i < length; i++)
        result[i] = k*result[i] + q;
    
    return result;
}
