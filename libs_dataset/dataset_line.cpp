#include "dataset_line.h"

#include <math.h>

DatasetLine::DatasetLine()
{
  width     = 8;
  height    = 8;
  channels  = 1;


  classes_count  = 5;
  unsigned int training_count = 50000;
  unsigned int testing_count  = 5000;


  training.resize(classes_count);

  create(training_count, false);
  create(testing_count, true);

  print();

  for (unsigned int i = 0; i < 20; i++)
    print_testing_item(i);
}

DatasetLine::~DatasetLine()
{

}

void DatasetLine::create(unsigned int count, bool testing)
{
  sDatasetItem item;

  for (unsigned int i = 0; i < count; i++)
  {
    auto item = create_item();

    if (testing)
      add_testing(item);
    else
      add_training(item);
  }
}

sDatasetItem DatasetLine::create_item()
{
  sDatasetItem result;

  result.input.resize(width*height*channels);
  result.output.resize(classes_count);


  for (unsigned int i = 0; i < classes_count; i++)
    result.input[i] = 0.0;

  for (unsigned int i = 0; i < classes_count; i++)
    result.output[i] = 0.0;



  float PI    = 3.141592654;
  float r     = width;
  float theta = rnd(-PI/4.0, PI/4.0);

  float x0    = width/2.0;
  float y0    = 0.0;
  float x1    = x0 + r*sin(theta);
  float y1    = y0 + r*cos(theta);




  float dt = 0.01;
  for (float t = 0.0; t < 1.0; t+= dt)
  {
    float x = (x1 - x0)*t + x0;
    float y = (y1 - y0)*t + y0;

    set_input(result.input, ceil(x), ceil(y), 1.0);
    set_input(result.input, floor(x), floor(y), 1.0);
  }


  int angle = theta*360/(2.0*PI);




  return result;
}


float DatasetLine::rnd(float min, float max)
{
  float v = (rand()%10000000)/10000000.0;

  float result = v*(max - min) + min;
  return result;
}

void DatasetLine::set_input(std::vector<float> &input, int x, int y, float value)
{
  if ((x >= 0)&&(y >= 0)&&(x < width)&&(y < height))
  {
    unsigned int idx = y*width + x;
    input[idx] = 1.0;
  }
}
