#include <iostream>
#include <dataset_time_wave.h>
#include <classification_experiment.h>


int main()
{
    DatasetTimeWave dataset(8000, 64);


    ClassificationExperiment experiment(dataset, "net_0/");
    experiment.run();
 
    std::cout << "program done\n";

    return 0;
}
