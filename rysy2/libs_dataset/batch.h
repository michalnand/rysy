#ifndef _BATCH_H_
#define _BATCH_H_

#include <tensor.h>
#include <dataset_interface.h>

class Batch
{
    public:
        Batch(Shape input_shape, Shape output_shape, unsigned int batch_size = 1024);
        Batch(Batch& other);
        Batch(const Batch& other);

        virtual ~Batch();
        Batch& operator= (Batch& other);
        Batch& operator= (const Batch& other);

    protected:
        void copy(Batch& other);
        void copy(const Batch& other);

    public:
        void create(DatasetInterface &dataset);

        std::vector<Tensor>& get_input_all();
        std::vector<Tensor>& get_output_all();

    private:
        std::vector<Tensor> input, output;
};

#endif
