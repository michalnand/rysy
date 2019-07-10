#ifndef _DATASET_TEST_H_
#define _DATASET_TEST_H_

#include <dataset_interface.h>


#define DATASET_TEST_MODE_NORMAL          ((unsigned int)0)
#define DATASET_TEST_MODE_RECURRENT   ((unsigned int)1)

class DatasetTest : public DatasetInterface
{
    public:
        DatasetTest(unsigned int mode = DATASET_TEST_MODE_NORMAL);

        virtual ~DatasetTest();

    private:
        void create(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input, unsigned int count);


};

#endif
