#ifndef _DATASET_LINE_CAMERA_H_
#define _DATASET_LINE_CAMERA_H_

#include "dataset_interface.h"

class DatasetLineCamera: public DatasetInterface
{
    public:
        DatasetLineCamera(unsigned int sensors_count, unsigned int pixels_count = 128, float noise_level = 0.1);
        virtual ~DatasetLineCamera();

    private:
        void create(unsigned int items_count, unsigned int pixels_count, float testing_ratio, float noise_level);
        std::vector<float> signal(float center, unsigned int length, float noise_level);


};

#endif
