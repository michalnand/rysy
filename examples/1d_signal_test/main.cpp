#include <iostream>
#include <dataset_line_camera.h>
#include <classification_experiment.h>


int main()
{
    unsigned int sensors_count  = 8;
    unsigned int pixels_count   = 128;
    float noise_level           = 0.6;

    DatasetLineCamera dataset(sensors_count, pixels_count, noise_level);


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
