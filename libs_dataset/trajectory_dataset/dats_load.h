#ifndef _DATS_LOAD_H_
#define _DATS_LOAD_H_

#include "dat_load.h"

class DatsLoad
{
  private:

    std::vector<DatLoad> dat;

    sDatExtremes extremes;

    unsigned int dat_count, column_count, lines_count;


  public:
    // Default constructor
    DatsLoad();

    DatsLoad(std::string json_config_file_name);

    // Copy constructor
    DatsLoad(DatsLoad& other);

    // Copy constructor
    DatsLoad(const DatsLoad& other);

    // Destructor
    virtual ~DatsLoad();

    // Copy assignment operator
    DatsLoad& operator= (DatsLoad& other);

    // Copy assignment operator
    DatsLoad& operator= (const DatsLoad& other);

  protected:
    void copy(DatsLoad& other);
    void copy(const DatsLoad& other);

  protected:
    void load(std::string json_config_file_name);

  public:
    void print();

    float get(unsigned int dat_idx, unsigned int column_idx, unsigned int line_idx);

    unsigned int get_dat_count();
    unsigned int get_columns_count();
    unsigned int get_lines_count();


    float get_max();
    float get_max_column(unsigned int column);
    float get_min();
    float get_min_column(unsigned int column);

    void normalise_column(float min = 0.0, float max = 1.0);

  private:
    void compute_metric();
    void find_extreme();


};

#endif
