#ifndef _CLASSIFICATION_EXPERIMENT_H_
#define _CLASSIFICATION_EXPERIMENT_H_

#include <string>
#include <dataset_interface.h>
#include <cnn.h>

class ClassificationExperiment
{
    public:
        ClassificationExperiment(DatasetInterface &dataset, std::string experiment_dir, std::string network_config_file);
        virtual ~ClassificationExperiment();

        void run();

    private:
        std::string delimiter();

    private:
        DatasetInterface *dataset;

        std::string experiment_dir, network_config_file;


};

#endif
