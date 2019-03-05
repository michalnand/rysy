#include "autoencoder_compare.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>

AutoencoderCompare::AutoencoderCompare()
{
    clear();
}

AutoencoderCompare::~AutoencoderCompare()
{

}


void AutoencoderCompare::clear()
{
    error_average_squared = 0.0;
    error_min_squared     = 1000000000.0;
    error_max_squared     = -error_min_squared;

    count = 0;
}


void AutoencoderCompare::compare(std::vector<float> &required_value, std::vector<float> &output_value)
{
    for (unsigned int i = 0; i < required_value.size(); i++)
    {
        float error = required_value[i] - output_value[i];
        float error_squared = pow(error, 2.0);

        error_average_squared+= error_squared;

        if (error_squared < error_min_squared)
            error_min_squared = error_squared;

        if (error_squared > error_max_squared)
            error_max_squared = error_squared;

        count++;
    }
}

void AutoencoderCompare::process(int fixed_bars_count)
{
    if (count == 0)
        return;

    error_average_squared = error_average_squared/count;
    json_result = process_json_result();
}

float AutoencoderCompare::get_error_average_squared()
{
  return error_average_squared;
}

float AutoencoderCompare::get_error_min_squared()
{
  return error_min_squared;
}

float AutoencoderCompare::get_error_max_squared()
{
  return error_max_squared;
}


void AutoencoderCompare::save_json_file(std::string json_file_name)
{
    JsonConfig json;
    json.result = json_result;

    json.save(json_file_name);
}

Json::Value AutoencoderCompare::process_json_result()
{
    Json::Value result;

    result["summary"]["count"]          = (unsigned int)count;
    result["summary"]["error_average"]  = error_average_squared;
    result["summary"]["error_min"]      = error_min_squared;
    result["summary"]["error_max"]      = error_max_squared;

    return result;
}
