#ifndef _CLASSIFICATION_EXPERIMENT_H_
#define _CLASSIFICATION_EXPERIMENT_H_

#include <string>
#include <dataset_interface.h>
#include <cnn.h>

#include <classification_compare.h>


class ClassificationExperiment
{
    public:
        ClassificationExperiment(DatasetInterface &dataset, std::string experiment_dir, std::string network_config_file);
        virtual ~ClassificationExperiment();

        void run();

    protected:
        virtual void process_best();

    private:
        std::string delimiter();

    protected:
        DatasetInterface *dataset;
        std::string experiment_dir, network_config_file;

    protected:
        ClassificationCompare compare_training, compare_training_top5;
        ClassificationCompare compare_testing, compare_testing_top5;
};

#endif
