#include <iostream>

#include <dataset_images.h>
#include <classification_experiment.h>
 
int main()
{
	srand(time(NULL));

	DatasetImages dataset("dataset.json");


	{
		ClassificationExperiment experiment(dataset, "net_0/");
		experiment.run();
	}

  std::cout << "program done\n";

  return 0;
}
