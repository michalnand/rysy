#include "time_series_dataset_interface.h"


TimeSeriesDatasetInterface::TimeSeriesDatasetInterface()
                           :DatasetInterface()
{

}

TimeSeriesDatasetInterface::~TimeSeriesDatasetInterface()
{

}


std::vector<sDatasetItem> TimeSeriesDatasetInterface::get_random_training_sequence()
{
    unsigned int class_idx = 0;
    do
    {
      class_idx = rand()%training.size();
    }
    while (training[class_idx].size() == 0);

    unsigned int item_idx = rand()%training[class_idx].size();

    return dataset_item_to_output(training[class_idx][item_idx]);
}

std::vector<sDatasetItem> TimeSeriesDatasetInterface::get_training_sequence(unsigned int class_idx, unsigned int idx)
{
    return dataset_item_to_output(training[class_idx][idx]);
}
std::vector<sDatasetItem> TimeSeriesDatasetInterface::get_testing_sequence(unsigned int idx)
{
    return dataset_item_to_output(testing[idx]);

}

std::vector<sDatasetItem> TimeSeriesDatasetInterface::get_random_testing_sequence()
{
    return dataset_item_to_output(testing[rand()%testing.size()]);
}



std::vector<sDatasetItem> TimeSeriesDatasetInterface::dataset_item_to_output(sDatasetItem &item)
{
    unsigned int size               = get_channels()*get_height()*get_width();
    unsigned int time_window_size   = item.input.size()/size;

    std::vector<sDatasetItem> result(time_window_size);


    unsigned int input_idx = 0;
    for (unsigned int t = 0; t < time_window_size; t++)
    {
            for (unsigned int i = 0; i < size; i++)
            {
                result[t].input[i] = item.input[input_idx];
                input_idx++;
            }

            result[t].output = item.output;
    }

    return result;
}
