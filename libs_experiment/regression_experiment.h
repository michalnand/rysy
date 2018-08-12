#ifndef _REGRESSION_EXPERIMENT_H_
#define _REGRESSION_EXPERIMENT_H_

#include <dataset_interface.h>
#include <batch.h>

#include <cnn.h>
#include <log.h>


struct sRegressionTestResult
{
  bool output_valid;
  bool too_big_difference;
  bool best_net;
  bool long_term_without_improvement;

  float error;
  float error_min;
  float error_max;
};

class RegressionExperiment
{
  protected:
    Json::Value       parameters;
    DatasetInterface  *dataset;
    Batch             *batch;

    sGeometry input_geometry, output_geometry;

  protected:
    std::string log_prefix;
    Log training_log;
    Log experiment_log;



  private:
    sRegressionTestResult result, best_result;

    unsigned int init_networks_try_count, epoch_count, epoch_without_improvement;
    float learning_rate, lambda, learning_rate_decay;

    float error_best;


  public:
    RegressionExperiment(std::string parameters_file_name, DatasetInterface *dataset);
    virtual ~RegressionExperiment();

    void run();

    sRegressionTestResult get_best_result()
    {
      return best_result;
    }

  private:
    void init();

  protected:
    virtual void best_net_log_process();

  private:
    int search_initial_net();
    CNN* load_saved_net(std::string best_net_file_name);


    sRegressionTestResult test(CNN *nn, bool quick_test = false);
    bool check_valid(std::vector<float> &v);

    float rms(std::vector<float> &va, std::vector<float> &vb);
    void print_vector(std::vector<float> &v);

  protected:
    virtual void fill_batch();
    void save_examples(CNN *nn);

};

#endif
