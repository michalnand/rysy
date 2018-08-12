#include "dataset_mnist.h"


DatasetMnist::DatasetMnist( std::string training_data_file_name, std::string training_labels_file_name,
                            std::string testing_data_file_name, std::string testing_labels_file_name,
                            unsigned int padding)
              :DatasetInterface()
{
  this->padding = padding;
  width     = 28 + 2*padding;
  height    = 28 + 2*padding;
  channels  = 1;

  load_dataset(&training, training_data_file_name, training_labels_file_name);
  load_dataset(&testing, testing_data_file_name, testing_labels_file_name);
}

DatasetMnist::~DatasetMnist()
{

}


int DatasetMnist::load_dataset(std::vector<struct sDatasetItem> *result, std::string data_file_name, std::string labels_file_name)
{
  unsigned int i, j;

  FILE *f_data, *f_labels;
  f_data = fopen(data_file_name.c_str(),"r");
  f_labels = fopen(labels_file_name.c_str(),"r");

  if (f_data == nullptr)
  {
    printf("data file %s opening error\n", data_file_name.c_str());
    return -1;
  }

  if (f_labels == nullptr)
  {
    printf("labels file %s opening error\n", labels_file_name.c_str());
    return -2;
  }

  unsigned int magic;
  magic = read_unsigned_int(f_data);
  if (magic != 0x00000803)
  {
    printf("data set magic error\n");
    return -3;
  }

  unsigned int items_count = read_unsigned_int(f_data);
  unsigned int rows = read_unsigned_int(f_data);
  unsigned int columns = read_unsigned_int(f_data);


  magic = read_unsigned_int(f_labels);
  if (magic != 0x00000801)
  {
    printf("label set magic error\n");
    return -4;
  }

  unsigned int labels_items_count = read_unsigned_int(f_labels);

  if (labels_items_count != items_count)
  {
    printf("unconsistent data and labels\n");
    return -5;
  }

  struct sDatasetItem item;

  item.input.resize(width*height*channels);
  item.output.resize(10);

  for (i = 0; i < item.input.size(); i++)
    item.input[i] = 0.0;

  for (i = 0; i < item.output.size(); i++)
    item.output[i] = 0.0;


  std::vector<float> raw_input;
  raw_input.resize(rows*columns);
  for (i = 0; i < raw_input.size(); i++)
    raw_input[i] = 0.0;

  for (j = 0; j < items_count; j++)
  {
    for (i = 0; i < raw_input.size(); i++)
    {
      unsigned char b = 0;
      int read_res = fread(&b, 1, sizeof(unsigned char), f_data);
      (void)read_res;

      raw_input[i] = b/256.0; //normalise into <0, 1> range
    }

    for (unsigned int y = 0; y < rows; y++)
      for (unsigned int x = 0; x < columns; x++)
        item.input[(y + padding)*width + (x + padding)] = raw_input[y*columns + x];

    for (i = 0; i < 10; i++)
      item.output[i] = 0.0;
 
    unsigned char b = 0;
    int read_res = fread(&b, 1, sizeof(unsigned char), f_labels);
    (void)read_res;
    item.output[b] =  1.0;

    result->push_back(item);
  }


  fclose(f_data);
  fclose(f_labels);

  printf("MNIST loading done %u [%u %u %u]\n", items_count, width, height, channels);

  return 0;
}

unsigned int DatasetMnist::read_unsigned_int(FILE *f)
{
  int read_res;

  unsigned char b0, b1, b2, b3;
  read_res = fread (&b3, 1, sizeof(unsigned char), f);
  read_res = fread (&b2, 1, sizeof(unsigned char), f);
  read_res = fread (&b1, 1, sizeof(unsigned char), f);
  read_res = fread (&b0, 1, sizeof(unsigned char), f);

  (void)read_res;

  unsigned int res = 0;
  res|= (unsigned int)b3 << 24;
  res|= (unsigned int)b2 << 16;
  res|= (unsigned int)b1 << 8;
  res|= (unsigned int)b0 << 0;

  return res;
}
