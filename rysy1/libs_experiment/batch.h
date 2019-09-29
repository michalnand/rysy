#ifndef _BATCH_H_
#define _BATCH_H_

#include <tensor.h>
#include <dataset_interface.h>

class Batch
{
    public:
        Batch();
        Batch(DatasetInterface &dataset, unsigned int batch_size = 65536);
        virtual ~Batch();

        void init(DatasetInterface &dataset, unsigned int batch_size = 65536);

    public:
        unsigned int size();
        void next();
        Tensor& get_input();
        Tensor& get_output();

        void fill_new();

    private:
        unsigned int batch_size, current_ptr;
        DatasetInterface *dataset;

        std::vector<Tensor> input;
        std::vector<Tensor> output;

};

#endif
