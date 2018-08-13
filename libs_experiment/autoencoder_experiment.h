#ifndef _AUTOENCODER_EXPERIMENT_H_
#define _AUTOENCODER_EXPERIMENT_H_


#include <dataset_interface.h>
#include <cnn.h>
#include <log.h>


class AutoencoderExperiment
{
  protected:
    DatasetInterface *dataset;
    std::string config_dir;

    sGeometry input_geometry;
    sGeometry output_geometry;

  protected:
    float rms_testing_average;
    float rms_testing_min    ;
    float rms_testing_max    ;

    float rms_training_average;
    float rms_training_min    ;
    float rms_training_max    ;

  protected:
    JsonConfig parameters;



  public:
    AutoencoderExperiment(DatasetInterface &dataset, std::string config_dir);
    virtual ~AutoencoderExperiment();

    void run();

  protected:
    void train_iterations(CNN &nn, unsigned int iterations);
    bool test(CNN &nn);

  protected:
    virtual void process_best(CNN &nn);
    void save_image(std::string file_name, std::vector<float> &v);


  private:
    bool is_valid(std::vector<float> &v);
    float compute_rms(std::vector<float> &va, std::vector<float> &vb);


};

#endif
