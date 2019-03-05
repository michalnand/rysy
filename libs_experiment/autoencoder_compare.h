#ifndef _AUTOENCODER_COMPARE_H_
#define _AUTOENCODER_COMPARE_H_


#include <string>
#include <vector>

#include "histogram.h"
#include "json_config.h"

class AutoencoderCompare
{
  private:
    unsigned long int count;

    double error_average_squared ;
    double error_min_squared     ;
    double error_max_squared     ;

    Json::Value json_result;

  public:
    AutoencoderCompare();
    virtual ~AutoencoderCompare();

    void clear();

    void compare(std::vector<float> &required_value, std::vector<float> &output_value);
    void process(int fixed_bars_count = -1);

    float get_error_average_squared();
    float get_error_min_squared();
    float get_error_max_squared();

    void save_json_file(std::string json_file_name);
    Json::Value process_json_result();
};

#endif
