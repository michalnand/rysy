#include <dataset_images.h>

#include <experimental/filesystem>
#include <image_load.h>
#include <thread>
#include <iostream>

DatasetImages::DatasetImages(std::string json_config_file_name)
              :DatasetInterface()
{
    JsonConfig json(json_config_file_name);

    grayscale   = json.result["grayscale"].asBool();
    unsigned int width      = json.result["width"].asInt();
    unsigned int height     = json.result["height"].asInt();

    max_items_per_folder    = json.result["max items per folder"].asInt();
    float testing_ratio     = json.result["testing ratio"].asFloat();

    unsigned int channels = 3;
    if (grayscale)
        channels  = 1;

    unsigned int classes_count = json.result["classes count"].asInt();


    input_shape.set(width, height, channels);
    output_shape.set(1, 1, classes_count);


    if (testing_ratio > 0.001)
    {
        load(json.result["dataset"], classes_count, 4, testing_ratio);
    }
    else
    {
        std::cout << "loading training\n";
        load(json.result["dataset"], classes_count, 4, 0.0);
        std::cout << "loading testing\n";
        load(json.result["dataset testing"], classes_count, 4, 1.0);
    }

    normalise_input();

    print();
}

DatasetImages::~DatasetImages()
{

}



void DatasetImages::load(Json::Value parameters, unsigned int classes_count, unsigned int load_threads_count, float testing_ratio)
{
    unsigned int ptr = 0;
    while (ptr < parameters.size())
    {
        std::vector<std::thread> load_threads;

        for (unsigned int i = 0; i < load_threads_count; i++)
        {
            if (ptr < parameters.size())
            {
                std::string path        = parameters[ptr]["path"].asString();
                unsigned int class_id   = parameters[ptr]["class"].asInt();

                load_threads.push_back(std::thread(&DatasetImages::load_dir, this, path, class_id, classes_count, testing_ratio));
                //load_dir(path, class_id, classes_count);

                ptr++;
            }
        }

        for (unsigned int i = 0; i < load_threads.size(); i++)
            load_threads[i].join();
    }
}

void DatasetImages::load_dir(std::string path, unsigned int class_id, unsigned int classes_count, float testing_ratio)
{
    std::cout << "loading directory " << path.c_str() << "\n";

    int items_count = 0;

    for (auto & p : std::experimental::filesystem::directory_iterator(path))
    {
        std::string image_file_name;
        image_file_name = p.path();

        auto extension = std::experimental::filesystem::path(image_file_name).extension();

        if ( (extension == ".png") || (extension == ".jpg"))
        {

            ImageLoad image(image_file_name, grayscale, false);

            if ((image.width() == input_shape.w()) && (image.width() == input_shape.h()))
            {
                std::vector<float> input = image.get();

                std::vector<float> output(classes_count);
                for (unsigned int i = 0; i < output.size(); i++)
                    output[i] = 0.0;
                output[class_id] = 1.0;

                float p = (rand()%100000)/100000.0;

                mutex.lock();

                if (p < testing_ratio)
                {
                    add_testing(input, output);
                }
                else
                {
                    add_training(input, output);
                }

                mutex.unlock();

                items_count++;

                if (max_items_per_folder != -1)
                {
                    if (items_count > max_items_per_folder)
                        return;
                }
            }
        }
    }
}
