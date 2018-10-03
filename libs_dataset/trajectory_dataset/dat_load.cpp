#include "dat_load.h"

#include <iostream>
#include <fstream>
#include <sstream>

DatLoad::DatLoad()
{
  extremes.max         = 0.0;
  extremes.min         = 0.0;
}

DatLoad::DatLoad(std::string file_name)
{
  load(file_name);
  find_extreme();
}

DatLoad::DatLoad(DatLoad& other)
{
  copy(other);
}

DatLoad::DatLoad(const DatLoad& other)
{
  copy(other);
}

DatLoad::~DatLoad()
{

}

DatLoad& DatLoad::operator= (DatLoad& other)
{
  copy(other);

  return *this;
}

DatLoad& DatLoad::operator= (const DatLoad& other)
{
  copy(other);

  return *this;
}

void DatLoad::copy(DatLoad& other)
{
  values = other.values;
  extremes = other.extremes;
}

void DatLoad::copy(const DatLoad& other)
{
  values = other.values;
  extremes = other.extremes;
}


unsigned int DatLoad::get_columns_count()
{
  if (values.size() > 0)
    return values[0].size();
  else
    return 0;
}

unsigned int DatLoad::get_lines_count()
{
  return values.size();
}

float DatLoad::get(unsigned int column, unsigned int line)
{
  return values[line][column];
}


float DatLoad::get_max()
{
  return extremes.max;
}

float DatLoad::get_max_column(unsigned int column)
{
  return extremes.max_column[column];
}

float DatLoad::get_min()
{
  return extremes.min;
}

float DatLoad::get_min_column(unsigned int column)
{
  return extremes.min_column[column];
}


void DatLoad::normalise_per_column(float min, float max)
{
  for (unsigned int j = 0; j < get_columns_count(); j++)
    normalise_column(j, get_min_column(j), get_max_column(j), min, max);
}

void DatLoad::normalise(float min, float max)
{
  for (unsigned int j = 0; j < get_columns_count(); j++)
    normalise_column(j, get_min(), get_max(), min, max);
}

void DatLoad::normalise_column_kq(unsigned int column, float k, float q)
{
  for (unsigned int i = 0; i < values.size(); i++)
    values[i][column] = k*values[i][column] + q;
} 

void DatLoad::print(bool verbose)
{
  std::cout << "columns " << get_columns_count() << "\n";
  std::cout << "lines " << get_lines_count() << "\n";
  std::cout << "min " << get_min() << "\n";
  std::cout << "max " << get_max() << "\n";

  std::cout << "min ";
  for (unsigned int i = 0; i < get_columns_count(); i++)
    std::cout << get_min_column(i) << " ";
  std::cout << "\n";

  std::cout << "max ";
  for (unsigned int i = 0; i < get_columns_count(); i++)
    std::cout << get_max_column(i) << " ";
  std::cout << "\n";

  std::cout << "\ndata \n";
  unsigned int lines = 5;

  if (verbose)
    lines = get_lines_count();

  if (lines > get_lines_count())
    lines = get_lines_count();

  for (unsigned int j = 0; j < lines; j++)
  {
    for (unsigned int i = 0; i < get_columns_count(); i++)
      std::cout << values[j][i] << " ";
    std::cout << "\n";
  }
  std::cout << "\n";
}


void DatLoad::find_extreme()
{
  extremes.max = values[0][0];
  extremes.min = extremes.max;

  for (unsigned int j = 0; j < get_lines_count(); j++)
  for (unsigned int i = 0; i < get_columns_count(); i++)
  {
    if (values[j][i] > extremes.max)
      extremes.max = values[j][i];

    if (values[j][i] < extremes.min)
      extremes.min = values[j][i];
  }

  extremes.max_column.resize(get_columns_count());
  extremes.min_column.resize(get_columns_count());


  for (unsigned int j = 0; j < get_columns_count(); j++)
  {
    extremes.max_column[j] = values[0][j];
    extremes.min_column[j] = values[0][j];

    for (unsigned int i = 0; i < get_lines_count(); i++)
    {
      if (values[i][j] > extremes.max_column[j])
        extremes.max_column[j] = values[i][j];

      if (values[i][j] < extremes.min_column[j])
        extremes.min_column[j] = values[i][j];
    }
  }
}


void DatLoad::load(std::string file_name)
{
  for (unsigned int i = 0; i < values.size(); i++)
  {
    values[i].clear();
  }

  values.clear();


  std::ifstream file(file_name);

  if (file.is_open() != true)
  {
    std::cout << "ERROR, file " << file_name << " opening error\n";
    return;
  }

  std::string line;



  while (std::getline(file, line))
  {
    std::stringstream iss(line);

    float value;
    std::vector<float> line_f;
    while (iss >> value)
    {
      line_f.push_back(value);
    }

    values.push_back(line_f);
  }

  find_extreme();
}


void DatLoad::normalise_column(unsigned int column, float source_min, float source_max, float dest_min, float dest_max)
{
  float k = 0.0;
  float q = 0.0;

  if (source_max > source_min)
  {
    k = (dest_max - dest_min)/(source_max - source_min);
    q = dest_max - k*source_max;
  }

  for (unsigned int i = 0; i < values.size(); i++)
    values[i][column] = k*values[i][column] + q;
}
