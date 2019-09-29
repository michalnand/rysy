#ifndef _REGRESSION_EXPERIMENT_H_
#define _REGRESSION_EXPERIMENT_H_


#include <dataset_interface.h>
#include <cnn.h>
#include <log.h>

#include "regression_compare.h"

class RegressionExperiment
{
  protected:
    RegressionCompare compare_testing, compare_training;

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
