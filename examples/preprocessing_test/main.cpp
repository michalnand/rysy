#include <iostream>
#include <preprocessing.h>

#include <image_load.h>
#include <image_save.h>

#include <timer.h>


void process(PreprocessingLayer &layer, Tensor &output, Tensor &input)
{
  layer.process(output, input);
}

int main()
{
  srand(time(NULL));

  JsonConfig json_config("preprocessing.json");

  ImageLoad input("input.jpg", false, true);
 
  sGeometry input_geometry;

  input_geometry.w = input.width();
  input_geometry.h = input.height();
  input_geometry.d = input.channels();

  Tensor input_tensor(input_geometry);
  Tensor output_tensor(input_geometry);

  input_tensor.set_from_host(input.get());


  Preprocessing preprocessing(json_config.result["preprocessing"], input_geometry);
  std::cout << "init done\n";

  preprocessing.process(output_tensor, input_tensor);


  std::vector<float> output_v(input.width()*input.height()*input.channels());
  output_tensor.set_to_host(output_v);
  ImageSave output(input.width(), input.height(), false);
  output.save("output.jpg", output_v);

  std::cout << "program done\n";

  return 0;
}
