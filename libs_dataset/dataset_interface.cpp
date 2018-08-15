#include "dataset_interface.h"

#include <log.h>

DatasetInterface::DatasetInterface()
{
  width     = 0;
  height    = 0;
  channels  = 0;
}

DatasetInterface::~DatasetInterface()
{

}

unsigned int DatasetInterface::get_training_size()
{
  return training.size();
}

unsigned int DatasetInterface::get_testing_size()
{
  return testing.size();
}

unsigned int DatasetInterface::get_unlabeled_size()
{
  if (unlabeled.size() > 0)
    return unlabeled.size();
  else
    return training.size();
}

unsigned int DatasetInterface::get_input_size()
{
  return width*height*channels;
}

unsigned int DatasetInterface::get_output_size()
{
  return training[0].output.size();
}

unsigned int DatasetInterface::get_width()
{
  return width;
}

unsigned int DatasetInterface::get_height()
{
  return height;
}

unsigned int DatasetInterface::get_channels()
{
  return channels;
}

sDatasetItem DatasetInterface::get_training(unsigned int idx)
{
    return training[idx];
}

sDatasetItem DatasetInterface::get_random_training()
{
    return training[rand()%get_training_size()];
}

sDatasetItem DatasetInterface::get_testing(unsigned int idx)
{
    return testing[idx];
}

sDatasetItem DatasetInterface::get_random_testing()
{
  return testing[rand()%get_testing_size()];
}

sDatasetItem DatasetInterface::get_unlabeled(unsigned int idx)
{
    if (unlabeled.size() > 0)
      return unlabeled[idx];
    else
      return training[idx];
}

sDatasetItem DatasetInterface::get_random_unlabeled()
{
  if (unlabeled.size() > 0)
    return unlabeled[rand()%unlabeled.size()];
  else
    return training[rand()%training.size()];
}

sDatasetItem DatasetInterface::get_random_training(float noise)
{
  sDatasetItem result = get_random_training();

  for (unsigned int i = 0; i < result.input.size(); i++)
    result.input[i] = (1.0 - noise)*result.input[i] + noise*rnd();

  return result;
}

sDatasetItem DatasetInterface::get_random_unlabeled(float noise)
{
  sDatasetItem result = get_random_unlabeled();

  for (unsigned int i = 0; i < result.input.size(); i++)
    result.input[i] = (1.0 - noise)*result.input[i] + noise*rnd();

  return result;
}


void DatasetInterface::print_training_item(unsigned int idx)
{
  printf("\n");
  for (unsigned int i = 0; i < training[idx].output.size(); i++)
    printf("%6.4f ", training[idx].output[i]);
  printf(" :\n");

  unsigned int ptr = 0;
  for (unsigned int ch = 0; ch < channels; ch++)
  {
    for (unsigned int y = 0; y < height; y++)
    {
      for (unsigned int x = 0; x < width; x++)
      {
        float v = training[idx].input[ptr];
	/*
        if (v > 0.01)
          printf("* ");
        else
          printf(". ");
       	*/
	printf("%6.4f ", v);

        ptr++;
      }
      printf("\n");
    }
    printf("\n");
  }

}


void DatasetInterface::print_testing_item(unsigned int idx)
{
  printf("\n");
  for (unsigned int i = 0; i < testing[idx].output.size(); i++)
    printf("%6.4f ", testing[idx].output[i]);
  printf(" :\n");

  unsigned int ptr = 0;
  for (unsigned int ch = 0; ch < channels; ch++)
  {
    for (unsigned int y = 0; y < height; y++)
    {
      for (unsigned int x = 0; x < width; x++)
      {
        float v = testing[idx].input[ptr];
	/*
        if (v > 0.01)
          printf("* ");
        else
          printf(". ");
	*/
        printf("%6.4f ", v);

        ptr++;
      }
      printf("\n");
    }
    printf("\n");
  }
}

unsigned int DatasetInterface::compare_biggest(unsigned int idx, char *output)
{
  unsigned int req_max_idx = 0;
  float req_max = -100000000.0;

  unsigned int test_max_idx = 1;
  float test_max = -100000000.0;

  for (unsigned int i = 0; i < testing[idx].output.size(); i++)
  {
    if (testing[idx].output[i] > req_max)
    {
      req_max = testing[idx].output[i];
      req_max_idx = i;
    }

    if (output[i] > test_max)
    {
      test_max = output[i];
      test_max_idx = i;
    }
  }

  if (req_max_idx == test_max_idx)
    return 1;

  return 0;
}

float DatasetInterface::rnd()
{
  float rndf = (rand()%100000)/100000.0;

  return rndf*2.0 - 1.0;
}

void DatasetInterface::export_h_testing(std::string file_name, unsigned int count)
{
  Log result(file_name);

  result << "#ifndef _DATASET_H_\n";
  result << "#define _DATASET_H_\n";
  result << "\n";

  result << "#define DATASET_COUNT (unsigned int)(" << count << ")\n";
  result << "\n";

  result << "#define DATASET_WIDTH  (unsigned int)(" << width << ")\n";
  result << "#define DATASET_HEIGHT  (unsigned int)(" << height << ")\n";
  result << "#define DATASET_CHANNELS  (unsigned int)(" << channels << ")\n";
  result << "\n";
  result << "\n";

  result << "const unsigned char dataset[]={";

  for (unsigned int j = 0; j < count; j++)
  {
    for (unsigned int i = 0; i < get_input_size(); i++)
    {
      if ((i%16) == 0)
        result << "\n";
      unsigned char v = testing[j].input[i]*255;
      result << v << ", ";
    }

    result << "\n";
  }
  result << "};\n\n\n";


  result << "const unsigned int labels[]={";

  for (unsigned int j = 0; j < count; j++)
  {
    if ((j%16) == 0)
      result << "\n";
    result << argmax(testing[j].output) << ", ";
  }
  result << "};\n\n\n";




  result << "#endif\n";
}

unsigned int DatasetInterface::argmax(std::vector<float> &v)
{
  unsigned int result = 0;
  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] > v[result])
      result = i;

  return result;
}


void DatasetInterface::shuffle()
{
  shuffle(training);
  shuffle(testing);
}

void DatasetInterface::shuffle(std::vector<sDatasetItem> &items)
{
  for (unsigned int i = 0; i < items.size(); i++)
  {
    unsigned int idx = rand()%items.size();

    sDatasetItem tmp  = items[i];
    items[i]          = items[idx];
    items[idx]        = tmp;
  }
}

void DatasetInterface::normalise(std::vector<float> &v, float min, float max)
{
  float v_max = v[0];
  float v_min = v[0];

  for (unsigned int i = 0; i < v.size(); i++)
  {
    if (v[i] > v_max)
      v_max = v[i];
    if (v[i] < v_min)
      v_min = v[i];
  }

  float k = 0.0;
  float q = 0.0;
  if (v_max > v_min)
  {
    k = (max - min)/(v_max - v_min);
    q = max - k*v_max;
  }

  for (unsigned int i = 0; i < v.size(); i++)
    v[i] = k*v[i] + q;
}

void DatasetInterface::compute_histogram()
{
  histogram.resize(get_output_size());

  for (unsigned int i = 0; i < histogram.size(); i ++)
    histogram[i] = 0;

  for (unsigned int i = 0; i < training.size(); i ++)
  {
    unsigned int class_idx = argmax(training[i].output);
    histogram[class_idx]++;
  }

  unsigned int histogram_max_idx   = 0;
  for (unsigned int i = 0; i < histogram.size(); i++)
    if (histogram[i] > histogram[histogram_max_idx])
      histogram_max_idx = i;

  histogram_max_count = histogram[histogram_max_idx];

  unsigned long int average = 0;

  for (unsigned int i = 0; i < histogram.size(); i++)
    average+= histogram[i];

  histogram_average_count = average / histogram.size();
}

void DatasetInterface::print_histogram()
{
  printf("\nhistogram : ");
  for (unsigned int i = 0; i < histogram.size(); i++)
    printf("%u ", histogram[i]);
  printf("\n");
}

void DatasetInterface::balance(float max_growth, bool use_average)
{
  shuffle();
  compute_histogram();
  //print_histogram();

  std::vector<int> required_add_count(histogram.size());

  if (use_average)
  {
    for (unsigned int i = 0; i < required_add_count.size(); i++)
      required_add_count[i] = histogram_average_count - histogram[i];
  }
  else
  {
    for (unsigned int i = 0; i < required_add_count.size(); i++)
      required_add_count[i] = histogram_max_count - histogram[i];
  }

  unsigned int max_dataset_size = training.size()*(1.0 + max_growth);

  while ((is_negative(required_add_count) == false) && (training.size() < max_dataset_size))
  {
    unsigned int dataset_size = training.size();
    unsigned int size_tmp = dataset_size;
    for (unsigned int i = 0; i < dataset_size; i++)
    {
      unsigned int class_idx = argmax(training[i].output);
      if (required_add_count[class_idx] > 0)
      {
        required_add_count[class_idx]--;
        size_tmp++;
        training.push_back(training[i]);

        if (size_tmp > max_dataset_size)
          break;
      }
    }
  }

  compute_histogram();
  // print_histogram();
}

bool DatasetInterface::is_negative(std::vector<int> &v)
{
  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] > 0)
      return false;

  return true;
}
