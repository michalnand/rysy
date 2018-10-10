#include "dataset_preprocessing.h"
#include <iostream>

DatasetPreprocessing::DatasetPreprocessing(DatasetInterface &dataset, std::string config_file_name)
              :DatasetInterface()
{
  width     = dataset.get_width();
  height    = dataset.get_height();
  channels  = dataset.get_channels();

  output_size = dataset.get_output_size();

  sGeometry input_geometry;

  input_geometry.w = width;
  input_geometry.h = height;
  input_geometry.d = channels;

  JsonConfig json(config_file_name);

  preprocessing.init(json.result, input_geometry);

  unsigned int augumentation_count = json.result["preprocessing"]["augumentation count"].asInt();


  //copy testing
  for (unsigned int i = 0; i < dataset.get_testing_size(); i++)
    testing.push_back(dataset.get_testing(i));

  //copy unlabeled
  for (unsigned int i = 0; i < dataset.get_unlabeled_size(); i++)
    unlabeled.push_back(dataset.get_unlabeled(i));



  //process training

  training.resize(output_size);

  sDatasetItem preprocessing_input, preprocessing_output;
  Tensor t_output, t_input;

  t_input.init(input_geometry);
  t_output.init(input_geometry);


  for (unsigned int j = 0; j < output_size; j++)
  {
    for (unsigned int i = 0; i < dataset.get_class_items_count(j); i++)
    {
      preprocessing_input  = dataset.get_training(j, i);
      preprocessing_output = preprocessing_input;

      //copy original
      add_training(preprocessing_input);

      //create artifical data
      t_input.set_from_host(preprocessing_input.input);
      for (unsigned int a = 0; a < augumentation_count; a++)
      {
        preprocessing.process(t_output, t_input, a);

        t_output.set_to_host(preprocessing_output.input);

        add_training(preprocessing_output);
      }
    }
  }


  print();
}

DatasetPreprocessing::~DatasetPreprocessing()
{

}
