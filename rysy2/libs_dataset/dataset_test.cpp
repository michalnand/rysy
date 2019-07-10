#include <dataset_test.h>

DatasetTest::DatasetTest(unsigned int mode)
{
    unsigned int classes_count  = 8;
    unsigned int training_count = 50000;
    unsigned int testing_count  = 2000;

    unsigned int sequence_legth  = classes_count*2;

    if (mode == DATASET_TEST_MODE_NORMAL)
        set_input_shape(Shape(sequence_legth, 1, 1, 1));
    if (mode == DATASET_TEST_MODE_RECURRENT)
        set_input_shape(Shape(1, 1, 1, sequence_legth));

    set_output_shape(Shape(1, 1, classes_count));


    create(training_output, training_input, training_count);
    create(testing_output, testing_input, testing_count);

    print();
}

DatasetTest::~DatasetTest()
{

}

void DatasetTest::create(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input, unsigned int count)
{
    output.resize(count);
    input.resize(count);

    std::vector<float> input_item(input_shape.size());
    std::vector<float> output_item(output_shape.size());

    for (unsigned int item = 0; item < count; item++)
    {
        unsigned int value = rand()%output_shape.size();

        for (unsigned int i = 0; i < input_item.size(); i++)
            input_item[i] = 0.0;

        for (unsigned int i = 0; i < value; i++)
            input_item[i] = 1.0;

        //shuffle
        for (unsigned int i = 0; i < input_item.size(); i++)
        {
            unsigned int idx_a = i;
            unsigned int idx_b = rand()%input_item.size();

            float tmp = input_item[idx_a];
            input_item[idx_a] = input_item[idx_b];
            input_item[idx_b] = tmp;
        }



        for (unsigned int i = 0; i < output_item.size(); i++)
            output_item[i] = 0.0;
        output_item[value] = 1.0;

        input[item]  = input_item;
        output[item] = output_item;
    }
}
