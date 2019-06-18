#ifndef _DATASET_ITEM_H_
#define _DATASET_ITEM_H_

#include <tensor.h>

class DatasetItem
{
    public:
        DatasetItem();

        DatasetItem(DatasetItem& other);
        DatasetItem(const DatasetItem& other);

        DatasetItem(Shape input_shape, Shape output_shape,
                    unsigned int input_time_steps = 1, unsigned int output_time_steps = 1);

        DatasetItem(    unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                        unsigned int output_width, unsigned int output_height, unsigned int output_depth,
                        unsigned int input_time_steps = 1,
                        unsigned int output_time_steps = 1);


        virtual ~DatasetItem();

        DatasetItem& operator= (DatasetItem& other);
        DatasetItem& operator= (const DatasetItem& other);

    protected:
        void copy(DatasetItem& other);
        void copy(const DatasetItem& other);


    public:
        void init(  Shape input_shape,
                    Shape output_shape,
                    unsigned int input_time_steps = 1,
                    unsigned int output_time_steps = 1);

        unsigned int get_input_time_steps();
        unsigned int get_output_time_steps();

        Shape get_input_shape();
        Shape get_output_shape();

        Tensor& get_input(unsigned int time_step = 0);
        Tensor& get_output(unsigned int time_step = 0);

        void set_input(std::vector<std::vector<float>> &input);
        void set_input(std::vector<float> &input, unsigned int time_step = 0);

        void set_output(std::vector<std::vector<float>> &output);
        void set_output(std::vector<float> &output, unsigned int time_step = 0);

    public:
        void print(bool full = false);

        void _set_random();


    private:
        Shape input_shape, output_shape;

        std::vector<Tensor> input;
        std::vector<Tensor> output;
};

#endif
