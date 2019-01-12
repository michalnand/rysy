#ifndef _TIME_SERIES_DATASET_INTERFACE_H_
#define _TIME_SERIES_DATASET_INTERFACE_H_

#include <dataset_interface.h>

class TimeSeriesDatasetInterface: public DatasetInterface
{
  public:
    TimeSeriesDatasetInterface();
    virtual ~TimeSeriesDatasetInterface();

  public:
      std::vector<sDatasetItem> get_random_training_sequence();
      std::vector<sDatasetItem> get_training_sequence(unsigned int class_idx, unsigned int idx);
      std::vector<sDatasetItem> get_testing_sequence(unsigned int idx);
      std::vector<sDatasetItem> get_random_testing_sequence();

  private:
      std::vector<sDatasetItem> dataset_item_to_output(sDatasetItem &item);
};

#endif
