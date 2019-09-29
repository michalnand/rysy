#include <iostream>
#include <dataset_line_camera.h>
#include <dataset_signal.h>
#include <classification_experiment.h>


int main()
{
    float noise_level           = 0.4;

    unsigned int classes  = 5;
    unsigned int length   = 256;
    unsigned int channels = 3;

    DatasetSignal dataset(classes, length, channels, noise_level);

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
