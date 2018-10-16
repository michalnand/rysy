#ifndef _REGRESSION_EXPERIMENT_H_
#define _REGRESSION_EXPERIMENT_H_


#include <dataset_interface.h>
#include <cnn.h>
#include <log.h>

#include "classification_compare.h"

class RegressionExperiment
{
  protected:
    ClassificationCompare compare_testing, compare_training;
    ClassificationCompare compare_testing_top5, compare_training_top5;

  protected:
    DatasetInterface *dataset;
     std::string config_dir;

  public:
    RegressionExperiment(DatasetInterface &dataset, std::string config_dir);
    virtual ~RegressionExperiment();

    void run();

  protected:
    void train_iterations(CNN &nn, unsigned int iterations);
    bool test(CNN &nn);

  protected:
    virtual void process_best();

  private:
    bool is_valid(std::vector<float> &v);


};

#endif
