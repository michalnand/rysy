#include "regression_compare.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>




RegressionCompare::RegressionCompare()
{
  clear();
}

RegressionCompare::RegressionCompare(unsigned int output_size)
{
  set_output_size(output_size);
}

RegressionCompare::~RegressionCompare()
{

}


void RegressionCompare::clear()
{
  output_size = 0;

  for (unsigned int i = 0; i < output_size; i++)
  {
    h_required[i].clear();
    h_resulted[i].clear();
    h_error[i].clear();
  }

  h_required.clear();
  h_resulted.clear();
  h_error.clear();

  h_required_summary.clear();
  h_resulted_summary.clear();
  h_error_summary.clear();
  h_error_squared.clear();

  error_average_squared = 0.0;
  error_min_squared     = 1000000000.0*output_size;
  error_max_squared     = -error_min_squared;

  count = 0;
}

void RegressionCompare::set_output_size(unsigned int output_size)
{
  clear();

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
  h_error_squared.clear();
}

void RegressionCompare::compare(std::vector<float> &required_value, std::vector<float> &output_value)
{
  for (unsigned int i = 0; i < output_size; i++)
  {
    h_required[i].add(required_value[i]);
    h_resulted[i].add(output_value[i]);

    float error = required_value[i] - output_value[i];
    float error_squared = pow(error, 2.0);

    h_error[i].add(error);

    h_required_summary.add(required_value[i]);
    h_resulted_summary.add(output_value[i]);
    h_error_summary.add(error);
    h_error_squared.add(error_squared);

    error_average_squared+= error_squared;

    if (error_squared < error_min_squared)
      error_min_squared = error_squared;

    if (error_squared > error_max_squared)
      error_max_squared = error_squared;

    count++;
  }
}

void RegressionCompare::process(int fixed_bars_count)
{
  //std::cout << "RegressionCompare::process()\n";

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

  /*
  std::cout << "bars count " << bars_count << "\n";
  std::cout << "output size " << output_size << "\n";
  std::cout << "count " << count << "\n";

  std::cout << "h_required " << h_required.size() << "\n";
  std::cout << "h_resulted " << h_resulted.size() << "\n";
  std::cout << "h_error " << h_error.size() << "\n";

  std::cout << "computing 1\n";
  */
  for (unsigned int i = 0; i < output_size; i++)
  {
    h_required[i].compute(bars_count);
    h_resulted[i].compute(bars_count);
    h_error[i].compute(bars_count);
  }

  h_required_summary.compute(bars_count);
  h_resulted_summary.compute(bars_count);
  h_error_summary.compute(bars_count);
  h_error_squared.compute(bars_count);

  error_average_squared = error_average_squared/count;

  json_result = process_json_result();
}

float RegressionCompare::get_error_average_squared()
{
  return error_average_squared;
}

float RegressionCompare::get_error_min_squared()
{
  return error_min_squared;
}

float RegressionCompare::get_error_max_squared()
{
  return error_max_squared;
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
  h_error_squared.save(log_file_name_prefix + "h_error_squared" + ".log");

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
  result["summary"]["error_average"]  = error_average_squared;
  result["summary"]["error_min"]      = error_min_squared;
  result["summary"]["error_max"]      = error_max_squared;

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

  for (unsigned int i = 0; i < h_error_squared.get_count(); i++)
  {
    result["summary"]["h_error_squared"][i]["value"] = h_error_squared.get(i).value;
    result["summary"]["h_error_squared"][i]["count"] = h_error_squared.get(i).count;
    result["summary"]["h_error_squared"][i]["normalised_count"] = h_error_squared.get(i).normalised_count;
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
