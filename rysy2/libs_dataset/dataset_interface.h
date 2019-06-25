#ifndef _DATASET_INTERFACE_H_
#define _DATASET_INTERFACE_H_

#include <vector>
#include <string>
#include <shape.h>

class DatasetInterface
{
    public:
        DatasetInterface();
        DatasetInterface(DatasetInterface& other);
        DatasetInterface(const DatasetInterface& other);

        DatasetInterface(Shape input_shape, Shape output_shape);


        virtual ~DatasetInterface();

        DatasetInterface& operator= (DatasetInterface& other);
        DatasetInterface& operator= (const DatasetInterface& other);

    protected:
        void copy(DatasetInterface& other);
        void copy(const DatasetInterface& other);


    public:
        Shape get_input_shape();
        Shape get_output_shape();

        unsigned int get_training_count();
        unsigned int get_testing_count();

        unsigned int get_classes_count();

    public:
        virtual void set_training_idx(unsigned int idx);
        virtual void set_random_training_idx();

        virtual std::vector<float>& get_training_input();
        virtual std::vector<float>& get_training_output();

        virtual std::vector<std::vector<float>>& get_training_input_all();
        virtual std::vector<std::vector<float>>& get_training_output_all();

    public:
        virtual std::vector<float>& get_testing_input(unsigned int idx);
        virtual std::vector<float>& get_testing_output(unsigned int idx);

        virtual std::vector<std::vector<float>>& get_testing_input_all();
        virtual std::vector<std::vector<float>>& get_testing_output_all();

    public:
        void set_input_shape(Shape input_shape);
        void set_output_shape(Shape output_shape);

        void add_training(std::vector<float>& input, std::vector<float> &output);
        void add_testing(std::vector<float>& input, std::vector<float> &output);

    public:
        void print();

    public:
        void clear();

    protected:
        void normalise_mat(std::vector<std::vector<float>> &mat);

    public:
        void normalise_input();
        void normalise_output();

    protected:
        Shape input_shape, output_shape;
        unsigned int current_training_idx;

        std::vector<std::vector<float>> training_input, training_output;
        std::vector<std::vector<float>> testing_input, testing_output;
};

#endif
