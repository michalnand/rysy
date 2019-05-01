#include <iostream>
#include <dataset_signal.h>
#include <classification_experiment.h>


int main()
{
    unsigned int classes_count = 10;
    unsigned int length = 128;
    float noise_level = 0.5;
    DatasetSignal dataset(classes_count, length, noise_level);


    JsonConfig parameters("experiments.json");

    for (unsigned int i = 0; i < parameters.result["experiments"].size(); i++)
    {
      std::string config_dir = parameters.result["experiments"][i].asString();
      ClassificationExperiment experiment(dataset, config_dir);
      experiment.run();
    }

    std::cout << "program done\n";
    return 0;
}
