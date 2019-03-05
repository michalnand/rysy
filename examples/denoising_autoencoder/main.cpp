#include <iostream>
#include <dataset_denoising_autoencoder.h>

#include <denoising_autoencoder_experiment.h>

#include <export_stl10.h>

int main()
{
	srand(time(NULL));


	/*
	DatasetDenoisingAutoencoder dataset("dataset.json");

	dataset.save_examples("net_0/examples/");
	dataset.save_examples("net_1/examples/");


	{
		DenoisingAutoencoderExperiment experiment(dataset, "net_0/");
		experiment.run();
	}


	{
    	DenoisingAutoencoderExperiment experiment(dataset, "net_1/");
    	experiment.run();
	}
	*/


	DatasetDenoisingAutoencoder dataset("dataset_images.json");

	dataset.save_examples("net_2/examples/");
	dataset.save_examples("net_3/examples/");

	{
		DenoisingAutoencoderExperiment experiment(dataset, "net_2/");
		experiment.run();
	}

	{
		DenoisingAutoencoderExperiment experiment(dataset, "net_3/");
		experiment.run();
	}

	/*
	ExportSTL10 stl_export("/home/michal/Downloads/stl10_binary/unlabeled_X.bin", "/home/michal/dataset/images/");
	stl_export.process(30000);
	*/
 

    std::cout << "program done\n";

    return 0;
}
