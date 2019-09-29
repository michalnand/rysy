#include <iostream>
#include <dataset_mnist.h>

#include <dataset_images.h>
#include <dataset_binary.h>
#include <classification_experiment.h>

int main()
{
	srand(time(NULL));

	/*
	DatasetImages dataset("dataset.json");

	dataset.save_to_binary("mnist_9_9/training.bin", "mnist_9_9/testing.bin", "mnist_9_9/unlabeled.bin");
	*/

	DatasetBinary dataset("mnist_9_9/training.bin", "mnist_9_9/testing.bin");


	{
    	ClassificationExperiment experiment(dataset, "net_0/");
    	experiment.run();
	}
	

    std::cout << "program done\n";

    return 0;
}
