#include "dataset_denoising_autoencoder.h"
#include <experimental/filesystem>
#include <image_load.h>
#include <image_save.h>
#include <thread>
#include <iostream>

DatasetDenoisingAutoencoder::DatasetDenoisingAutoencoder(std::string json_config_file_name)
{
    JsonConfig json(json_config_file_name);

    grayscale     = json.result["grayscale"].asBool();
    width         = json.result["width"].asInt();
    height        = json.result["height"].asInt();

    max_items_per_folder = json.result["max items per folder"].asInt();
    float testing_ratio = json.result["testing ratio"].asFloat();

    white_noise_level           = json.result["white noise level"].asFloat();
    salt_and_peper_noise_level  = json.result["salt and peper noise level"].asFloat();
    color_noise_level           = json.result["color noise level"].asFloat();


    if (grayscale)
        channels  = 1;
    else
        channels  = 3;

    training.resize(1);
    output_size = width*height*channels;

    if (testing_ratio > 0.001)
        load(json.result["dataset"], 4, testing_ratio);
    else
    {
        std::cout << "loading training\n";
        load(json.result["dataset"], 4, 0.0);
        std::cout << "loading testing\n";
        load(json.result["dataset testing"], 4, 1.0);
    }


  print();
}

DatasetDenoisingAutoencoder::~DatasetDenoisingAutoencoder()
{

}

void DatasetDenoisingAutoencoder::save_examples(std::string path, unsigned int count)
{
    for (unsigned int i = 0; i < count; i++)
    {
        ImageSave image(width, height, grayscale);

        std::string original_file_name = path + std::to_string(i) + "_0_original.png";
        std::string noised_file_name   = path + std::to_string(i) + "_1_noised.png";

        image.save(noised_file_name, testing[i].input);
        image.save(original_file_name, testing[i].output);
    }
}



void DatasetDenoisingAutoencoder::load(Json::Value parameters, unsigned int load_threads_count, float testing_ratio)
{
    unsigned int ptr = 0;

    while (ptr < parameters.size())
    {
        std::vector<std::thread> load_threads;

        for (unsigned int i = 0; i < load_threads_count; i++)
        {
            if (ptr < parameters.size())
            {
                std::string path = parameters[ptr]["path"].asString();
                load_threads.push_back(std::thread(&DatasetDenoisingAutoencoder::load_dir, this, path, testing_ratio));
                ptr++;
            }
        }

        for (unsigned int i = 0; i < load_threads.size(); i++)
            load_threads[i].join();
    }
}

void DatasetDenoisingAutoencoder::load_dir(std::string path, float testing_ratio)
{
    std::cout << "loading directory " << path.c_str() << "\n";


  int items_count = 0;

  for (auto & p : std::experimental::filesystem::directory_iterator(path))
   {
     std::string image_file_name;
     image_file_name = p.path();


     if (std::experimental::filesystem::path(image_file_name).extension() == ".png")
     {
         std::cout << "loading file " << image_file_name << "\n";
       ImageLoad image(image_file_name, grayscale, true);

       if ((image.width() == width) && (image.width() == height))
       {
           sDatasetItem item;

           item.output = image.get();
           item.input  = apply_noise(item.output);

           float p = (rand()%100000)/100000.0;

           mutex.lock();

           if (p < testing_ratio)
                add_testing(item);
           else
           {
                training[0].push_back(item);
                training_size++;
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


std::vector<float> DatasetDenoisingAutoencoder::apply_noise(std::vector<float> &input)
{
    std::vector<float> result = input;

    normalise(result, -1.0, 1.0);


    if (grayscale)
    {
        float gr_noise = rnd(-1.0, 1.0)*color_noise_level;

        for (unsigned int i = 0; i < result.size(); i++)
            result[i]+= gr_noise;
    }
    else
    {
        float r_noise = rnd(-1.0, 1.0)*color_noise_level;
        float g_noise = rnd(-1.0, 1.0)*color_noise_level;
        float b_noise = rnd(-1.0, 1.0)*color_noise_level;

        unsigned int layer_size = width*height;

        for (unsigned int i = 0; i < layer_size; i++)
        {
            result[i + layer_size*0]+= r_noise;
            result[i + layer_size*1]+= g_noise;
            result[i + layer_size*2]+= b_noise;
        }
    }

    for (unsigned int i = 0; i < result.size(); i++)
    {
        result[i]+= white_noise_level*rnd(-1.0, 1.0);

        if (rnd(0, 1) < salt_and_peper_noise_level)
            result[i] = srnd();
    }

    /*
    if ((rand()%2) == 0)
    if (grayscale == false)
    {
        unsigned int layer_size = width*height;

        for (unsigned int i = 0; i < layer_size; i++)
        {
            float r = result[i + layer_size*0];
            float g = result[i + layer_size*1];
            float b = result[i + layer_size*2];

            float res = 0.3*r + 0.59*g + 0.11*b;

            result[i + layer_size*0] = res;
            result[i + layer_size*1] = res;
            result[i + layer_size*2] = res;
        }
    }
    */
   
    normalise(result, 0.0, 1.0);

    return result;
}


float DatasetDenoisingAutoencoder::rnd(float min, float max)
{
    float v = (rand()%1000000)/1000000.0;

    return (max - min)*v + min;
}

float DatasetDenoisingAutoencoder::srnd()
{
    float result = 0.0;

        if ((rand()%2) == 0)
            result = -1.0;
        else
            result =  1.0;

    return result;
}

void DatasetDenoisingAutoencoder::normalise(std::vector<float> &v, float min, float max)
{
  float max_v = v[0];
  float min_v = v[0];
  for (unsigned int i = 0; i < v.size(); i++)
  {
    if (v[i] > max_v)
      max_v = v[i];

    if (v[i] < min_v)
      min_v = v[i];
  }

  float k = 0.0;
  float q = 0.0;

  if (max_v > min_v)
  {
    k = (max - min)/(max_v - min_v);
    q = max - k*max_v;
  }

  for (unsigned int i = 0; i < v.size(); i++)
  {
    v[i] = k*v[i] + q;
  }
}
