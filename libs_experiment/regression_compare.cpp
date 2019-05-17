#include "regression_compare.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>




RegressionCompare::RegressionCompare()
{
    set_output_size(0);
}

RegressionCompare::RegressionCompare(unsigned int output_size)
{
    set_output_size(output_size);
}

RegressionCompare::~RegressionCompare()
{

}

void RegressionCompare::set_output_size(unsigned int output_size)
{
    this->output_size = output_size;

    h_required.resize(output_size);
    h_resulted.resize(output_size);
    h_error.resize(output_size);

    for (unsigned int i = 0; i < output_size; i++)
    {
        h_required[i].clear();
        h_resulted[i].clear();
        h_error[i].clear();
    }

    h_required_summary.clear();
    h_resulted_summary.clear();
    h_error_summary.clear();
    h_error_summary_euclidean.clear();

    count = 0;
}

void RegressionCompare::compare(std::vector<float> &required_value, std::vector<float> &output_value)
{
    for (unsigned int i = 0; i < output_size; i++)
    {
        h_required[i].add(required_value[i]);
        h_resulted[i].add(output_value[i]);

        float error = required_value[i] - output_value[i];
        h_error[i].add(error);
    }

    for (unsigned int i = 0; i < output_size; i++)
    {
        h_required_summary.add(required_value[i]);
        h_resulted_summary.add(output_value[i]);

        float error = required_value[i] - output_value[i];
        h_error_summary.add(error);
    }


    float error_sum = 0.0;
    for (unsigned int i = 0; i < output_size; i++)
    {
        float error_squared = pow(required_value[i] - output_value[i], 2.0);
        error_sum+= error_squared;
    }

    error_sum = sqrt(error_sum);
    h_error_summary_euclidean.add(error_sum);

    count++;
}



void RegressionCompare::process(int fixed_bars_count)
{
    if (count == 0)
        return;

    int bars_count;

    if (fixed_bars_count > 0)
    {
        bars_count = fixed_bars_count;
    }
    else
    {
        bars_count = count/100;

        if (bars_count > 500)
            bars_count = 500;

        if (bars_count < 50)
            bars_count = 50;

        if ((bars_count%2) == 0)
            bars_count+= 1;
    }


  for (unsigned int i = 0; i < output_size; i++)
  {
      h_required[i].compute(bars_count);
      h_resulted[i].compute(bars_count);
      h_error[i].compute(bars_count);
  }

  h_required_summary.compute(bars_count);
  h_resulted_summary.compute(bars_count);
  h_error_summary.compute(bars_count);
  h_error_summary_euclidean.compute(bars_count);

  json_result = process_json_result();
}

float RegressionCompare::get_error_average()
{
    return h_error_summary_euclidean.get_average();
}

float RegressionCompare::get_error_std()
{
    return h_error_summary_euclidean.get_std();
}


void RegressionCompare::save_text_file(std::string log_file_name_prefix)
{
    for (unsigned int i = 0; i < output_size; i++)
    {
        h_required[i].save(log_file_name_prefix + "h_required_" + std::to_string(i) + ".log");
        h_resulted[i].save(log_file_name_prefix + "h_resulted_" + std::to_string(i) + ".log");
        h_error[i].save(log_file_name_prefix + "h_error_" + std::to_string(i) + ".log");
    }

    h_required_summary.save(log_file_name_prefix + "h_required_summary" + ".log");
    h_resulted_summary.save(log_file_name_prefix + "h_resulted_summary" + ".log");
    h_error_summary.save(log_file_name_prefix + "h_error_summary" + ".log");
    h_error_summary_euclidean.save(log_file_name_prefix + "h_error_summary_euclidean" + ".log");
}

void RegressionCompare::save_json_file(std::string json_file_name)
{
  JsonConfig json;
  json.result = json_result;

  json.save(json_file_name);
}

Json::Value RegressionCompare::process_json_result()
{
    Json::Value result;

    result["summary"]["count"]          = count;
    result["summary"]["error_average"]  = h_error_summary_euclidean.get_average();
    result["summary"]["error_std"]      = h_error_summary_euclidean.get_std();

    for (unsigned int i = 0; i < h_required_summary.get_count(); i++)
    {
        result["summary"]["h_required_summary"][i]["value"] = h_required_summary.get(i).value;
        result["summary"]["h_required_summary"][i]["count"] = h_required_summary.get(i).count;
        result["summary"]["h_required_summary"][i]["normalised_count"] = h_required_summary.get(i).normalised_count;
    }

    for (unsigned int i = 0; i < h_resulted_summary.get_count(); i++)
    {
        result["summary"]["h_resulted_summary"][i]["value"] = h_resulted_summary.get(i).value;
        result["summary"]["h_resulted_summary"][i]["count"] = h_resulted_summary.get(i).count;
        result["summary"]["h_resulted_summary"][i]["normalised_count"] = h_resulted_summary.get(i).normalised_count;
    }

    for (unsigned int i = 0; i < h_error_summary.get_count(); i++)
    {
        result["summary"]["h_error_summary"][i]["value"] = h_error_summary.get(i).value;
        result["summary"]["h_error_summary"][i]["count"] = h_error_summary.get(i).count;
        result["summary"]["h_error_summary"][i]["normalised_count"] = h_error_summary.get(i).normalised_count;
    }

    for (unsigned int i = 0; i < h_error_summary_euclidean.get_count(); i++)
    {
        result["summary"]["h_error_summary_euclidean"][i]["value"] = h_error_summary_euclidean.get(i).value;
        result["summary"]["h_error_summary_euclidean"][i]["count"] = h_error_summary_euclidean.get(i).count;
        result["summary"]["h_error_summary_euclidean"][i]["normalised_count"] = h_error_summary_euclidean.get(i).normalised_count;
    }

    for (unsigned int j = 0; j < output_size; j++)
    for (unsigned int i = 0; i < h_error_summary.get_count(); i++)
    {
        result["detailed"][j]["h_required"][i]["value"] = h_required[j].get(i).value;
        result["detailed"][j]["h_required"][i]["count"] = h_required[j].get(i).count;
        result["detailed"][j]["h_required"][i]["normalised_count"] = h_required[j].get(i).normalised_count;

        result["detailed"][j]["h_resulted"][i]["value"] = h_resulted[j].get(i).value;
        result["detailed"][j]["h_resulted"][i]["count"] = h_resulted[j].get(i).count;
        result["detailed"][j]["h_resulted"][i]["normalised_count"] = h_resulted[j].get(i).normalised_count;

        result["detailed"][j]["h_error"][i]["value"] = h_error[j].get(i).value;
        result["detailed"][j]["h_error"][i]["count"] = h_error[j].get(i).count;
        result["detailed"][j]["h_error"][i]["normalised_count"] = h_error[j].get(i).normalised_count;
    }

  return result;
}
