#include "classification_compare.h"

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

ClassificationCompare::ClassificationCompare()
{
  succes_result = 0.0;
  this->top_n_count = 1;
}

ClassificationCompare::ClassificationCompare(unsigned int classes_count_, unsigned int top_n_count_)
{
  classes_count = classes_count_;
  succes_result = 0.0;
  this->top_n_count = top_n_count_;

  set_classes_count(classes_count);
}

ClassificationCompare::~ClassificationCompare()
{

}

void ClassificationCompare::compare(std::vector<float> &required_value, std::vector<float> &output_value)
{
  unsigned int required_idx = argmax(required_value, classes_count);

  auto result = get_top_n(output_value, top_n_count);

  bool correct = false;

  for (unsigned int i = 0; i < result.size(); i++)
  {
    unsigned int network_output_idx = result[i];

    if (required_idx == network_output_idx)
    {
      confusion_matrix[network_output_idx][required_idx]++;
      correct = true;
      break;
    }
  }

  if (!correct)
    confusion_matrix[result[0]][required_idx]++;


}


void ClassificationCompare::set_classes_count(unsigned int classes_count)
{
  this->classes_count = classes_count;

  confusion_matrix.resize(classes_count);

  for (unsigned int j = 0; j < classes_count; j++)
  {
    confusion_matrix[j].resize(classes_count);

    for (unsigned int i = 0; i < classes_count; i++)
    confusion_matrix[j][i] = 0;
  }


  for (unsigned int j = 0; j < classes_count; j++)
    for (unsigned int i = 0; i < classes_count; i++)
      confusion_matrix[j][i] = 0;

  class_success_rate.resize(confusion_matrix.size());
}

void ClassificationCompare::set_top_n_count(unsigned int top_n_count_)
{
  this->top_n_count = top_n_count_;
}


void ClassificationCompare::clear()
{
  for (unsigned int j = 0; j < classes_count; j++)
    for (unsigned int i = 0; i < classes_count; i++)
      confusion_matrix[j][i] = 0;

  for (unsigned int j = 0; j < class_success_rate.size(); j++)
    class_success_rate[j] = 0.0;

  sum_all = 0;
  sum_correct = 0;
  succes_result = 0;

  for (unsigned int j = 0; j < summary_result.size(); j++)
    summary_result[j] = 0.0;
}

float ClassificationCompare::process(bool percent_result)
{
  float k = 1.0;
  if (percent_result)
    k = 100.0;

  succes_result = 0.0;

  sum_all = 0;
  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
    for (unsigned int i = 0; i < confusion_matrix.size(); i++)
      sum_all+= confusion_matrix[j][i];

  sum_correct = 0;
  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
      sum_correct+= confusion_matrix[j][j];



  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
  {
    unsigned int sum_column = 0;
    for (unsigned int i = 0; i < confusion_matrix.size(); i++)
      sum_column+= confusion_matrix[i][j];

    if (sum_column > 0)
      class_success_rate[j] = (confusion_matrix[j][j]*k)/sum_column;
    else
      class_success_rate[j] = 0.0;
  }

  succes_result = (sum_correct*k)/sum_all;

  summary_result = process_summary_result();
  text_result = process_text_result();
  json_result = process_json_result();

  return succes_result;
}

float ClassificationCompare::get_success()
{
  return succes_result;
}

std::vector<float> ClassificationCompare::process_summary_result()
{
  std::vector<float> result;

  result.push_back(sum_correct);
  result.push_back(sum_all - sum_correct);
  result.push_back(succes_result);

  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
    for (unsigned int i = 0; i < confusion_matrix.size(); i++)
      result.push_back(confusion_matrix[j][i]);

  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
      result.push_back(class_success_rate[j]);


  return result;
}

std::string ClassificationCompare::process_text_result()
{
  std::ostringstream result;

  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
  {
    for (unsigned int i = 0; i < confusion_matrix.size(); i++)
      result << std::right << std::setw(9) << confusion_matrix[j][i];
    result << "\n";
  }
  result << "\n";


  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
    result << std::right << std::setw(9) << std::setprecision(5) << class_success_rate[j];
  result << std::right << std::setw(9) <<  "\n\n\n";

  result << std::right << std::setw(9) <<  sum_correct;
  result << std::right << std::setw(9) <<  sum_all - sum_correct;
  result << std::right << std::setw(9) <<  succes_result << "%";
  result << std::right << std::setw(9) <<   "\n";

  return result.str();
}

Json::Value ClassificationCompare::process_json_result()
{
  Json::Value result;

  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
    for (unsigned int i = 0; i < confusion_matrix.size(); i++)
      result["confusion_matrix"][j][i] = confusion_matrix[j][i];

  for (unsigned int j = 0; j < confusion_matrix.size(); j++)
    result["class_success_rate"][j] = class_success_rate[j];

  result["total_correct"] = sum_correct;
  result["total_wrong"] = sum_all - sum_correct;
  result["succes_result"] = succes_result;

  return result;
}

void ClassificationCompare::save_text_file(std::string file_name)
{
  std::ofstream result_file(file_name);
  result_file << process_text_result();
  result_file.close();
}

void ClassificationCompare::save_json_file(std::string file_name)
{
  std::ofstream result_file(file_name);
  result_file << process_json_result();
  result_file.close();
}

unsigned int ClassificationCompare::argmax(std::vector<float> &v, unsigned int size)
{
  unsigned int result = 0;
  for (unsigned int i = 0; i < size; i++)
    if (v[i] > v[result])
      result = i;

  return result;
}


std::vector<unsigned int> ClassificationCompare::get_top_n(std::vector<float> &confidence, unsigned int top_n_count)
{
  std::vector<unsigned int> result;

  for (unsigned int j = 0; j < top_n_count; j++)
  {
    float max = -1000000000.0;
    unsigned int max_idx = 0;

    for (unsigned int i = 0; i < confidence.size(); i++)
      if (confidence[i] > max)
      if (not_in(result, i))
      {
        max = confidence[i];
        max_idx = i;
      }

    result.push_back(max_idx);
  }

  return result;
}

bool ClassificationCompare::not_in(std::vector<unsigned int> &vect, unsigned int value)
{
  for (unsigned int i = 0; i < vect.size(); i++)
    if (value == vect[i])
      return false;

  return true;
}
