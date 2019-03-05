#ifndef _DATASET_DENOISING_AUTOENCODER_H_
#define _DATASET_DENOISING_AUTOENCODER_H_

#include "dataset_interface.h"

#include <json_config.h>
#include <mutex>

class DatasetDenoisingAutoencoder: public DatasetInterface
{
    private:
        bool grayscale;
        int max_items_per_folder;

        std::mutex mutex;

    private:
        float white_noise_level, salt_and_peper_noise_level;
        float color_noise_level;

    public:
        DatasetDenoisingAutoencoder(std::string json_config_file_name);
        ~DatasetDenoisingAutoencoder();

        void save_examples(std::string path, unsigned int count = 32);

    private:
        void load(Json::Value parameters, unsigned int load_threads_count = 4, float testing_ratio = 0.1);
        void load_dir(std::string path, float testing_ratio);

        std::vector<float> apply_noise(std::vector<float> &input);

        float rnd(float min, float max);
        float srnd();

        void normalise(std::vector<float> &v, float min, float max);
};

#endif
