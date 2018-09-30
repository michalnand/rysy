#include "dataset_pair.h"


DatasetPair::DatasetPair(DatasetInterface &dataset, int training_size_, int testing_size_)
              :DatasetInterface()
{
  width     = dataset.get_width();
  height    = dataset.get_height();
  channels  = dataset.get_channels()*2;

  if (training_size_ > 0)
    training_size = training_size_;
  else
    training_size = dataset.get_training_size();

  if (testing_size_ > 0)
    testing_size = testing_size_;
  else
    testing_size  = dataset.get_testing_size();

  output_size = 2;

  training.resize(output_size);

  create(dataset, training_size, false);
  create(dataset, testing_size, true);

  print();
}

DatasetPair::~DatasetPair()
{

}


void DatasetPair::create(DatasetInterface &dataset, unsigned int count, bool set_testing)
{
  sDatasetItem item;

  item.input.resize(get_input_size());
  item.output.resize(get_output_size());

  for (unsigned int n = 0; n < count; n++)
  {
    bool equal;

    if ((rand()%2) == 0)
      equal = true;
    else
      equal = false;

    sDatasetItem item_a, item_b;

    if (set_testing)
      item_a = dataset.get_random_testing();
    else
      item_a = dataset.get_random_training();


    if (equal)
    {
      do
      {
        if (set_testing)
          item_b = dataset.get_random_testing();
        else
          item_b = dataset.get_random_training();
      }
      while (argmax(item_a.output) != argmax(item_b.output));
    }
    else
    {
      do
      {
        if (set_testing)
          item_b = dataset.get_random_testing();
        else
          item_b = dataset.get_random_training();
      }
      while (argmax(item_a.output) == argmax(item_b.output));
    }

    if (equal)
    {
      item.output[0] = 1.0;
      item.output[1] = 0.0;
    }
    else
    {
      item.output[0] = 0.0;
      item.output[1] = 1.0;
    }

    unsigned int ptr = 0;
    for (unsigned int i = 0; i < item_a.input.size(); i++)
    {
      item.input[ptr] = item_a.input[i];
      ptr++;
    }

    for (unsigned int i = 0; i < item_b.input.size(); i++)
    {
      item.input[ptr] = item_b.input[i];
      ptr++;
    }

    if (set_testing)
      testing.push_back(item);
    else
    {
      if (equal)
        training[0].push_back(item);
      else
        training[1].push_back(item);
    }
  }
}
