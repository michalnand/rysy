#ifndef _DATASET_SIGNAL_H_
#define _DATASET_SIGNAL_H_

#include "dataset_interface.h"

class DatasetSignal: public DatasetInterface
{
    public:
        DatasetSignal(unsigned int classes_count, unsigned int length = 128, unsigned int channels = 1, float noise_level = 0.1);
        virtual ~DatasetSignal();

    private:
        void create(unsigned int items_count, unsigned int length, float testing_ratio, float noise_level);
        std::vector<float> signal(float frequency, unsigned int length, unsigned int channels, float noise_level);


};

#endif
