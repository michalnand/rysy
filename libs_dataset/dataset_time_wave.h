#ifndef _DATASET_TIME_WAVE_H_
#define _DATASET_TIME_WAVE_H_

#include <time_series_dataset_interface.h>

class DatasetTimeWave: public TimeSeriesDatasetInterface
{
    private:
        unsigned int waves_count;
        std::vector<float> a, b, c, d;

    public:
        DatasetTimeWave(unsigned int items_count, unsigned int time_window_size, bool rnn_mode = false);
        virtual ~DatasetTimeWave();

    private:
        sDatasetItem create_item(unsigned int time_window_size);
        float func(float x);
};

#endif
