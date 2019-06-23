#ifndef _REGRESSION_EXPERIMENT_H_
#define _REGRESSION_EXPERIMENT_H_

#include <string>
#include <dataset_interface.h>
#include <cnn.h>

class RegressionExperiment
{
    public:
        RegressionExperiment(DatasetInterface &dataset, std::string experiment_dir, std::string network_config_file);
        virtual ~RegressionExperiment();

        void run();

    protected:
        void process_best(CNN &cnn);

    private:
        std::string delimiter();

    protected:
        DatasetInterface *dataset;
        std::string experiment_dir, network_config_file;
};

#endif
