#include "dataset_tic_tac_toe.h"
#include <fstream>

DatasetTicTacToe::DatasetTicTacToe(std::string data_file_name, float testing_ratio, unsigned int padding)
{
  width      = 3 + 2*padding;
  height     = 3 + 2*padding;
  channels   = 2;

  output_size = 2;
  training.resize(output_size);

  load_dataset(data_file_name, testing_ratio, padding);

  print();
}


DatasetTicTacToe::~DatasetTicTacToe()
{

}

int DatasetTicTacToe::load_dataset(std::string data_file_name, float testing_ratio, unsigned int padding)
{
  sDatasetItem item;

  item.input.resize(width*height*channels);
  item.output.resize(2);

  std::ifstream file(data_file_name);

  std::string line;


  while (std::getline(file, line))
  {
    float p = (rand()%100000)/100000.0;

    for (unsigned int rotation = 0; rotation < 8; rotation++)
    {
      for (unsigned int i = 0; i < item.input.size(); i++)
        item.input[i] = 0.0;

      unsigned int str_ptr = 0;

      for (unsigned int y = 0; y < 3; y++)
      for (unsigned int x = 0; x < 3; x++)
      {
        int ch = -1;

        if (line[str_ptr] == 'x')
          ch = 0;
        else
        if (line[str_ptr] == 'o')
          ch = 1;


        str_ptr+= 2;

        unsigned int y_ = 0, x_ = 0;
        rotate(y_, x_, y, x, rotation);

        if (ch != -1)
          item.input[(ch*height + y_ + padding)*width + x_ + padding] = 1.0;
      }

      if (line[18] == 'p')
      {
        item.output[0] = 1.0;
        item.output[1] = 0.0;
      }
      else
      {
        item.output[0] = 0.0;
        item.output[1] = 1.0;
      }

      if (p < testing_ratio)
        add_testing(item);
      else
        add_training(item);
    }
  }



  return 0.0;
}


void DatasetTicTacToe::rotate(unsigned int &y_, unsigned int &x_, unsigned int &y, unsigned int &x, unsigned int rotation)
{
  unsigned int size = 3;

  switch (rotation)
  {
    case 0:
            y_ = y;
            x_ = x;
            break;

    case 1:
            y_ = size - 1 - y;
            x_ = x;
            break;

    case 2:
            y_ = y;
            x_ = size - 1 - x;
            break;

    case 3:
            y_ = size - 1 - y;
            x_ = size - 1 - x;
            break;

    case 4:
            y_ = x;
            x_ = y;
            break;

    case 5:
            y_ = size - 1 - x;
            x_ = y;
            break;

    case 6:
            y_ = x;
            x_ = size - 1 - y;
            break;

    case 7:
            y_ = size - 1 - x;
            x_ = size - 1 - y;
            break;
    }
}
