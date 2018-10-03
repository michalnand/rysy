#ifndef _DAT_LOAD_H_
#define _DAT_LOAD_H_

#include <vector>
#include <string>


struct sDatExtremes
{
  float max, min;
  std::vector<float> max_column, min_column;
};

class DatLoad
{
  public:
    // Default constructor
    DatLoad();

    DatLoad(std::string file_name);

    // Copy constructor
    DatLoad(DatLoad& other);

    // Copy constructor
    DatLoad(const DatLoad& other);

    // Destructor
    virtual ~DatLoad();

    // Copy assignment operator
    DatLoad& operator= (DatLoad& other);

    // Copy assignment operator
    DatLoad& operator= (const DatLoad& other);

  protected:
    void copy(DatLoad& other);
    void copy(const DatLoad& other);

  private:
    std::vector<std::vector<float>> values;

    sDatExtremes extremes;

  public:
    unsigned int get_columns_count();
    unsigned int get_lines_count();

    float get(unsigned int column, unsigned int line);

    float get_max();
    float get_max_column(unsigned int column);
    float get_min();
    float get_min_column(unsigned int column);

    void normalise_per_column(float min = 0.0, float max = 1.0);
    void normalise(float min = 0.0, float max = 1.0);
    void normalise_column_kq(unsigned int column, float k, float q);

    void print(bool verbose = false);

  public:
    void load(std::string file_name);
    void find_extreme();
 
  private:
    void normalise_column(unsigned int column, float source_min, float source_max, float dest_min, float dest_max);

};

#endif
