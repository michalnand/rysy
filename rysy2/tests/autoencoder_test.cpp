#include <regression_experiment.h>
#include <dataset_mnist.h>
#include <dataset_autoencoder.h>
#include <iostream>


int main()
{
    std::string dataset_path = "/home/michal/dataset/mnist/";

    DatasetMnist dataset_source(    dataset_path + "train-images.idx3-ubyte",
                                    dataset_path + "train-labels.idx1-ubyte",
                                    dataset_path + "t10k-images.idx3-ubyte",
                                    dataset_path + "t10k-labels.idx1-ubyte");

    DatasetAutoencoder dataset(dataset_source);

    dataset.print();

    RegressionExperiment experiment(dataset, "regression_experiment/", "network_config.json");
    experiment.run();

    std::cout << "program done\n";
}
