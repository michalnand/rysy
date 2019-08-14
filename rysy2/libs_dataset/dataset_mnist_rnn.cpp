#include <dataset_mnist_rnn.h>
#include <iostream>

DatasetMnistRNN::DatasetMnistRNN( std::string training_data_file_name, std::string training_labels_file_name,
                            std::string testing_data_file_name, std::string testing_labels_file_name)
                :DatasetInterface()
{
    set_input_shape(Shape(1, 1, 28, 28));
    set_output_shape(Shape(1, 1, 10));
 
    load(training_input, training_output, training_data_file_name, training_labels_file_name);
    load(testing_input, testing_output, testing_data_file_name, testing_labels_file_name);
    print();
}

DatasetMnistRNN::~DatasetMnistRNN()
{

}

int DatasetMnistRNN::load(    std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output,
                            std::string data_file_name, std::string labels_file_name)
{
    unsigned int i, j;

    FILE *f_data, *f_labels;
    f_data      = fopen(data_file_name.c_str(),"r");
    f_labels    = fopen(labels_file_name.c_str(),"r");

    if (f_data == nullptr)
    {
        std::cout << "data file " << data_file_name.c_str() << " ERROR\n";
        return -1;
    }

    if (f_labels == nullptr)
    {
        std::cout << "labels file " << labels_file_name.c_str() << " ERROR\n";
        return -2;
    }

    unsigned int magic;
    magic = read_unsigned_int(f_data);
    if (magic != 0x00000803)
    {
        std::cout << "data magic error\n";
        return -3;
    }

    unsigned int items_count = read_unsigned_int(f_data);
    unsigned int rows = read_unsigned_int(f_data);
    unsigned int columns = read_unsigned_int(f_data);

    magic = read_unsigned_int(f_labels);
    if (magic != 0x00000801)
    {
        std::cout << "labels magic error\n";
        return -4;
    }

    unsigned int labels_items_count = read_unsigned_int(f_labels);

    if (labels_items_count != items_count)
    {
        std::cout << "unconsistent data and labels\n";
        return -5;
    }

    std::vector<float> _input(input_shape.size());
    std::vector<float> _output(output_shape.size());

    for (i = 0; i < _input.size(); i++)
        _input[i] = 0.0;

    for (i = 0; i < _output.size(); i++)
        _output[i] = 0.0;

    std::vector<float> raw_input(input_shape.size());
    for (i = 0; i < raw_input.size(); i++)
        raw_input[i] = 0.0;

    for (j = 0; j < items_count; j++)
    {
        for (i = 0; i < raw_input.size(); i++)
        {
            unsigned char b = 0;
            int read_res = fread(&b, 1, sizeof(unsigned char), f_data);
            (void)read_res;

            raw_input[i] = b/256.0;
        }

        unsigned int ptr = 0;
        for (unsigned int y = 0; y < rows; y++)
        for (unsigned int x = 0; x < columns; x++)
        {
            _input[ptr] = raw_input[y*columns + x];
            ptr++;
        }

        for (i = 0; i < 10; i++)
            _output[i] = 0.0;

        unsigned char b = 0;
        int read_res = fread(&b, 1, sizeof(unsigned char), f_labels);
        (void)read_res;

        _output[b] =  1.0;

        input.push_back(_input);
        output.push_back(_output);
    }

    fclose(f_data);
    fclose(f_labels);

    std::cout << "loading MNIST done\n";

    return 0;
}

unsigned int DatasetMnistRNN::read_unsigned_int(FILE *f)
{
    int read_res;

    unsigned char b0, b1, b2, b3;
    read_res = fread (&b3, 1, sizeof(unsigned char), f);
    read_res = fread (&b2, 1, sizeof(unsigned char), f);
    read_res = fread (&b1, 1, sizeof(unsigned char), f);
    read_res = fread (&b0, 1, sizeof(unsigned char), f);

    (void)read_res;

    unsigned int res = 0;
    res|= (unsigned int)b3 << 24;
    res|= (unsigned int)b2 << 16;
    res|= (unsigned int)b1 << 8;
    res|= (unsigned int)b0 << 0;

    return res;
}
