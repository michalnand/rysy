#include "dataset_binary.h"
#include <fstream>


DatasetBinary::DatasetBinary(   std::string training_file_name,
                                std::string testing_file_name)
              :DatasetInterface()
{
    {
        std::ifstream file;
        file.open(training_file_name, std::ios::in | std::ios::binary);

        int testing_count = load_header(file);

        if (testing_count > 0)
        {
            training.resize(get_output_size());

            sDatasetItem item;
            item.input.resize(get_input_size());
            item.output.resize(get_output_size());

            for (unsigned int i = 0; i < (unsigned int)testing_count; i++)
            {
                load_item(file, item);
                add_training(item);
            }
        }

        file.close();
    }


    {
        std::ifstream file;
        file.open(testing_file_name, std::ios::in | std::ios::binary);

        int testing_count = load_header(file);

        if (testing_count > 0)
        { 
            sDatasetItem item;
            item.input.resize(get_input_size());
            item.output.resize(get_output_size());

            for (unsigned int i = 0; i < (unsigned int)testing_count; i++)
            {
                load_item(file, item);
                add_testing(item);
            }
        }

        file.close();
    }

    print();
}

DatasetBinary::~DatasetBinary()
{

}
