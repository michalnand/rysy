#include "dataset_stack.h"
#include <fstream>
#include <iostream>
#include <json_config.h>
#include <math.h>
#include <image.h>

DatasetStack::DatasetStack( unsigned int width,
                            unsigned int height,
                            unsigned int training_count,
                            unsigned int testing_count,
                            unsigned int unlabeled_count,
                            float noise,
                            float noise_background,
                            float alignment_probability)
              :DatasetInterface()
{
  this->width    = width;
  this->height   = height;
  this->channels = 2;
  this->noise   = noise;
  this->noise_background = noise_background;
  this->alignment_probability = alignment_probability;

  for (unsigned int i = 0; i < training_count; i++)
  {
    sDatasetItem item = create_item();
    add_training(item);
  }

  for (unsigned int i = 0; i < testing_count; i++)
  {

    std::string item_image_file_name = "testing/";
    item_image_file_name+=std::to_string(i);

    //testing.push_back(create_item(&item_image_file_name));

    sDatasetItem item = create_item();
    add_testing(item);
  }

  for (unsigned int i = 0; i < unlabeled_count; i++)
  {
    auto tmp = create_item();
    for (unsigned int k = 0; k < tmp.output.size(); k++)
      tmp.output[k] = 0.0;

    add_unlabeled(tmp);
  }

  print();
}

DatasetStack::~DatasetStack()
{

}

void DatasetStack::save(std::string file_name_prefix, std::vector<struct sDatasetItem> &items)
{
  JsonConfig json;

  json.result["width"]  = width;
  json.result["height"] = height;
  json.result["channels"] = 2;
  json.result["noise"] = noise;
  json.result["noise_background"] = noise_background;
  json.result["alignment_probability"] = alignment_probability;

  json.result["items_count"] = (unsigned int)items.size();
  json.result["binary_file_name"] = file_name_prefix + ".bin";


  json.save(file_name_prefix + ".json");

  FILE *f;
  f = fopen((file_name_prefix + ".bin").c_str(), "w");


  for (unsigned int j = 0; j < items.size(); j++)
  {
    for (unsigned int i = 0; i < items[j].input.size(); i++)
      fwrite(&items[j].input[i], sizeof(float), 1, f);

    for (unsigned int i = 0; i < items[j].output.size(); i++)
      fwrite(&items[j].output[i], sizeof(float), 1, f);
  }

  fclose(f);
}

sDatasetItem DatasetStack::create_item(std::string *item_image_file_name)
{
  sDatasetItem result;


  unsigned int item_max_w = width-2;
  unsigned int item_max_h = height-2;

  unsigned int item_w = 1 + rand()%(item_max_w-1);
  unsigned int item_h = 1 + rand()%(item_max_h-1);

/*
  unsigned int item_w = 6;
  unsigned int item_h = 6;
*/

  auto noise_mask      = get_noise_mask(item_w, item_h, noise);
  auto background_mask = get_noise_mask(width, height, 1.0 - noise_background);

  int x0 = rand()%(width - item_w);
  int y0 = rand()%(height - item_h);
  int x1 = 0;
  int y1 = 0;

  float v = fabs((rand()%100000) / 100000.0);

  if (v < alignment_probability)
  {
    x1 = x0;
    y1 = y0;
  }
  else
  {
    x1 = rand()%(width - item_w);
    y1 = rand()%(height - item_h);
  }

  auto rectangle_a = random_rectangle(item_w, item_h, x0, y0, noise_mask, background_mask);
  auto rectangle_b = random_rectangle(item_w, item_h, x1, y1, noise_mask, background_mask);

  float confidence = compute_confidence(rectangle_a, rectangle_b);


/*
  print_rectangle(rectangle_a);
  print_rectangle(rectangle_b);

  if (confidence > 0.999)
    printf("\nalignment with confidence %f\n", confidence);
  else
    printf("\nNO alignment with confidence %f\n", confidence);
  printf("\n\n********************************************\n\n");

*/
  for (unsigned int j = 0; j < rectangle_a.size(); j++)
    for (unsigned int i = 0; i < rectangle_a[j].size(); i++)
      result.input.push_back(rectangle_a[j][i]);

  for (unsigned int j = 0; j < rectangle_b.size(); j++)
    for (unsigned int i = 0; i < rectangle_b[j].size(); i++)
      result.input.push_back(rectangle_b[j][i]);

  if (confidence > 0.999)
  {
    result.output.push_back(-1.0);
    result.output.push_back(1.0);
  }
  else
  {
    result.output.push_back(1.0);
    result.output.push_back(-1.0);
  }


  if (item_image_file_name != nullptr)
  {
    std::string file_name;

    if (result.output[1] > 0.0)
      file_name= *item_image_file_name + "_1.png";
    else
      file_name= *item_image_file_name + "_0.png";

    save_image(file_name, rectangle_a, rectangle_b);
  }

  return result;
}

std::vector<std::vector<float>> DatasetStack::get_noise_mask(unsigned int rect_w, unsigned int rect_h, float noise)
{
  std::vector<std::vector<float>> result;

  result.resize(rect_h);
  for (unsigned int j = 0; j < result.size(); j++)
  {
    result[j].resize(rect_w);
    for (unsigned int i = 0; i < result[j].size(); i++)
      if ( ((rand()%1000)/1000.0) < noise )
        result[j][i] = 0.0;
      else
        result[j][i] = 1.0;
  }

  return result;
}

float gaussian_noise(float a)
{
  float tmp = fabs((rand()%100000) / 100000.0);

  return (1.0 - exp(-a*tmp));
}

int sgn_random()
{
  if (rand()%2)
    return 1;
  return -1;
}

std::vector<std::vector<float>> DatasetStack::random_rectangle( unsigned int rect_w,
                                                                unsigned int rect_h,
                                                                int x,
                                                                int y,
                                                                std::vector<std::vector<float>> &noise_mask,
                                                                std::vector<std::vector<float>> &background_mask)
{
  std::vector<std::vector<float>> result;

    if (x < 0)
      x = 0;
    if (x >= (int)(width - rect_w))
      x = (int)(width - rect_w)-1;

    if (y < 0)
      y = 0;
    if (y >= (int)(height - rect_h))
      y = (int)(height - rect_h)-1;


  result.resize(height);
  for (unsigned int j = 0; j < result.size(); j++)
  {
    result[j].resize(width);
    for (unsigned int i = 0; i < result[j].size(); i++)
      result[j][i] = background_mask[j][i];
  }

  for (unsigned int j = 0; j < rect_h; j++)
    for (unsigned int i = 0; i < rect_w; i++)
    {
      if (noise_mask[j][i] > 0.0)
        result[y + j][x + i] = 1.0;
    }



  return result;
}

void DatasetStack::print_rectangle(std::vector<std::vector<float>> &rect)
{
  for (unsigned int j = 0; j < rect.size(); j++)
  {
    for (unsigned int i = 0; i < rect[j].size(); i++)
    {
      if (rect[j][i] > 0.0)
        printf("* ");
      else
        printf(". ");
    }

    printf("\n");
  }

  printf("\n");
}


float DatasetStack::compute_confidence(std::vector<std::vector<float>> &rect_ref, std::vector<std::vector<float>> &rect_test)
{
  float result = 0.0;

  float sum_ref = 0.0;
  for (unsigned int j = 0; j < rect_ref.size(); j++)
    for (unsigned int i = 0; i < rect_ref[j].size(); i++)
      sum_ref+= rect_ref[j][i];

  if (sum_ref > 0.0)
  {
    float dot = 0.0;
    for (unsigned int j = 0; j < rect_ref.size(); j++)
      for (unsigned int i = 0; i < rect_ref[j].size(); i++)
      {
        dot+= rect_ref[j][i]*rect_test[j][i];
      }
    result = dot/sum_ref;
  }

  return result;
}

void DatasetStack::save_image(std::string item_image_file_name, std::vector<std::vector<float>> &rectangle_a, std::vector<std::vector<float>> &rectangle_b)
{
  Image image(width, height);

  for (unsigned int j = 0; j < height; j++)
    for (unsigned int i = 0; i < width; i++)
    {
      image.pixels[j][i].b[0] = rectangle_a[j][i];
      image.pixels[j][i].b[1] = rectangle_b[j][i];
      image.pixels[j][i].b[2] = 1.0;
    }

  image.save(item_image_file_name);
}
