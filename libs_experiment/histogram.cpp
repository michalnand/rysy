#include "histogram.h"

#include <iostream>
#include <fstream>


Histogram::Histogram()
{
  clear();
}

Histogram::~Histogram()
{

}

void Histogram::add(float value)
{
  values.push_back(value);
}

void Histogram::clear()
{
  min = 0.0;
  max = 0.0;
  average = 0.0;
  count = 0;

  values.clear();
  histogram.clear();
}


void Histogram::compute(unsigned int count)
{
  this->count = count;

  find_range();
  init_histogram();

  if (min >= max)
  {
    histogram[0].count = values.size();
  }
  else
  {
    for (unsigned int i = 0; i < values.size(); i++)
    {
      unsigned int idx = find_nearest(values[i]);
      histogram[idx].count++;
    }
  }

  unsigned int cnt = 0;
  for (unsigned int i = 0; i < histogram.size(); i++)
    cnt+= histogram[i].count;

  if (cnt > 0)
  {
    for (unsigned int i = 0; i < histogram.size(); i++)
      histogram[i].normalised_count = histogram[i].count*1.0/cnt;
  }

}


unsigned int Histogram::get_count()
{
  return count;
}

sHistogramItem Histogram::get(unsigned int idx)
{
  return histogram[idx];
}

std::vector<sHistogramItem> Histogram::get()
{
  return histogram;
}

std::string Histogram::asString()
{
  std::string result;

  for (unsigned int i = 0; i < histogram.size(); i++)
  {
    result+= std::to_string(i) + " ";
    result+= std::to_string(histogram[i].value) + " ";
    result+= std::to_string(histogram[i].count) + " ";
    result+= std::to_string(histogram[i].normalised_count) + "\n";
  }

  return result;
}

void Histogram::print()
{
  std::cout << asString() << "\n";
}

void Histogram::save(std::string file_name)
{
  std::ofstream file(file_name);
  file << asString();
  file.close();
}

float Histogram::get_max()
{
  return max;
}

float Histogram::get_min()
{
  return min;
}

float Histogram::get_average()
{
  return average;
}

void Histogram::find_range()
{
  min = values[0];
  max = min;

  for (unsigned int i = 0; i < values.size(); i++)
  {
    if (values[i] < min)
      min = values[i];

    if (values[i] > max)
      max = values[i];
  }

  float sum = 0.0;
  for (unsigned int i = 0; i < values.size(); i++)
    sum+= values[i];

  average = sum/values.size();
}

void Histogram::init_histogram()
{
  histogram.resize(count);

  float k = 0.0;
  float q = 0.0;

  if (max > min)
  {
    float tmp = histogram.size();
    k = (max - min)/tmp;
    q = min;
  }

  for (unsigned int i = 0; i < histogram.size(); i++)
  {
    histogram[i].count = 0;
    histogram[i].normalised_count = 0.0;
    histogram[i].value = k*i + q;
  }
}


unsigned int Histogram::find_nearest(float value)
{
  unsigned int center = 0;
  unsigned int left   = 0;
  unsigned int right  = count-1;

  unsigned int result = 0;

  if (value > histogram[0].value)
  {
    while (left <= right)
    {
      center = (left + right)/2;

      if (value < histogram[center].value)
        right = center - 1;
      else
        left  = center + 1;
    }

    result = (right + left)/2;
  }

  return result;
}
