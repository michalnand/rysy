#ifndef _CLASSIFICATION_EXPERIMENT_H_
#define _CLASSIFICATION_EXPERIMENT_H_


#include <dataset_interface.h>
#include <cnn.h>
#include <log.h>
#include <batch.h>

#include "classification_compare.h"
#include <preprocessing.h>

class ClassificationExperiment
{
    protected:
        ClassificationCompare compare_testing, compare_training;
        ClassificationCompare compare_testing_top5, compare_training_top5;

    protected:
        DatasetInterface *dataset;
        std::string config_dir;

    protected:
        Batch training_batch;

    public:
        ClassificationExperiment(DatasetInterface &dataset, std::string config_dir);
        virtual ~ClassificationExperiment();

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
