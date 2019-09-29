#ifndef _HISTOGRAM_H_
#define _HISTOGRAM_H_

#include <vector>
#include <string>

struct sHistogramItem
{
  unsigned int count;
  float normalised_count;
  float value;
};

class Histogram
{
  private:
    unsigned int count;
    float min, max, average, standard_deviation;

  private:
    std::vector<float> values;

    std::vector<sHistogramItem> histogram;

  public:
    Histogram();
    virtual ~Histogram();

    void add(float value);
    void clear();

    void compute(unsigned int count = 201);

  public:
    unsigned int get_count();
    sHistogramItem get(unsigned int idx);
    std::vector<sHistogramItem> get();
    std::string asString();

    void print();
    void save(std::string file_name);

  public:
    float get_max();
    float get_min();
    float get_average();
    float get_std();

  private:
    void compute_stats();
    void init_histogram();
    unsigned int find_nearest(float value);

};

#endif
