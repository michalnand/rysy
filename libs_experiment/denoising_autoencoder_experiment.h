#ifndef _DENOISING_AUTOENCODER_EXPERIMENT_H_
#define _DENOISING_AUTOENCODER_EXPERIMENT_H_


#include <dataset_interface.h>
#include <cnn.h>
#include <log.h>

#include "autoencoder_compare.h"

class DenoisingAutoencoderExperiment
{
    protected:
      AutoencoderCompare compare_testing, compare_training;
 
    protected:
      DatasetInterface *dataset;
       std::string config_dir;

    public:
      DenoisingAutoencoderExperiment(DatasetInterface &dataset, std::string config_dir);
      virtual ~DenoisingAutoencoderExperiment();

      void run();

    protected:
      void train_iterations(CNN &nn, unsigned int iterations);
      bool test(CNN &nn);

    protected:
      virtual void process_best();

    private:
      bool is_valid(std::vector<float> &v);

      void save_examples(CNN &nn, std::string path, unsigned int count = 32);
};

#endif
