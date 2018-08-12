#ifndef _AUTOENCODER_TRAIN_H_
#define _AUTOENCODER_TRAIN_H_

#include <batch.h>
#include <cnn.h>
#include <dataset_interface.h>
#include <log.h>

class AutoencoderTrain
{
  private:
    sGeometry input_geometry;
    sGeometry output_geometry;

    Batch *training_batch, *testing_batch;

    float rms, rms_min, rms_max, rms_best;
    bool nan_error;
    Log experiment_log;

    std::string experiment_path;

  public:
    AutoencoderTrain(DatasetInterface *dataset, std::string experiment_path);
    ~AutoencoderTrain();

    void main();

    void test_images(std::string result_path);
    void test_image(std::string input_file_name, std::string output_file_name);

    void export_all();

  protected:
    unsigned int output_transform(Tensor &tensor);

    void export_batch(std::string export_path, Batch *batch);

    void test(CNN &nn, bool quick);

  protected:
    void save_image(std::string file_name, Tensor &tensor);

};

#endif
