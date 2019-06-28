#include <classification_compare.h>
#include <math.h>

#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>

ClassificationCompare::ClassificationCompare()
{
    init(0);
}

ClassificationCompare::ClassificationCompare(ClassificationCompare& other)
{
    copy(other);
}

ClassificationCompare::ClassificationCompare(const ClassificationCompare& other)
{
    copy(other);
}

ClassificationCompare::ClassificationCompare(unsigned int classes_count, unsigned int top_n_count)
{
    init(classes_count, top_n_count);
}

ClassificationCompare::~ClassificationCompare()
{

}

ClassificationCompare& ClassificationCompare::operator= (ClassificationCompare& other)
{
    copy(other);
    return *this;
}

ClassificationCompare& ClassificationCompare::operator= (const ClassificationCompare& other)
{
    copy(other);
    return *this;
}


void ClassificationCompare::copy(ClassificationCompare& other)
{
    this->confusion_matrix  = other.confusion_matrix;
    this->success_count     = other.success_count;
    this->miss_count        = other.miss_count;
    this->total_count       = other.total_count;
    this->accuracy          = other.accuracy;
    this->nan_error         = other.nan_error;

    this->top_n_count       = other.top_n_count;
}

void ClassificationCompare::copy(const ClassificationCompare& other)
{
    this->confusion_matrix  = other.confusion_matrix;
    this->success_count     = other.success_count;
    this->miss_count        = other.miss_count;
    this->total_count       = other.total_count;
    this->accuracy          = other.accuracy;
    this->nan_error         = other.nan_error;

    this->top_n_count     = other.top_n_count;
}

void ClassificationCompare::init(unsigned int classes_count, unsigned int top_n_count)
{
    this->classes_count = classes_count;

    this->confusion_matrix.resize(this->classes_count);
    for (unsigned int j = 0; j < this->classes_count; j++)
    {
        this->confusion_matrix[j].resize(this->classes_count);
        for (unsigned int i = 0; i < this->classes_count; i++)
            this->confusion_matrix[j][i] = 0;
    }

    this->class_success.resize(this->classes_count);
    for (unsigned int j = 0; j < this->classes_count; j++)
        this->class_success[j] = 0.0;

    this->success_count   = 0;
    this->miss_count      = 0;
    this->total_count     = 0;

    this->accuracy        = 0;

    this->nan_error = false;

    this->top_n_count = top_n_count;
}

int ClassificationCompare::add(std::vector<float> &target_output, std::vector<float> &predicted_output)
{
    if (is_valid(target_output) == false)
    {
        this->nan_error = true;
        return -1;
    }

    if (is_valid(predicted_output) == false)
    {
        this->nan_error = true;
        return -2;
    }

    if (target_output.size() != predicted_output.size())
    {
        this->nan_error = true;
        return -3;
    }

    if (this->nan_error)
    {
        return -4;
    }

    if (top_n_count == 1)
    {
        unsigned int target     = class_idx(target_output);
        unsigned int predicted  = class_idx(predicted_output);

        confusion_matrix[predicted][target]++;
    }
    else
    {
        unsigned int target = class_idx(target_output);
        auto top_n_result   = get_top_n(predicted_output, top_n_count);

        bool correct = false;
        for (unsigned int i = 0; i < top_n_result.size(); i++)
        {
            unsigned int predicted = top_n_result[i];
            if (predicted == target)
            {
                confusion_matrix[predicted][target]++;
                correct = true;
                break;
            }
        }

        if (correct != true)
        {
            unsigned int predicted = top_n_result[0];
            confusion_matrix[predicted][target]++;
        }

    }

    return 0;
}

void ClassificationCompare::compute()
{
    float eps = 0.00000000001;

    this->accuracy    = 0;

    this->total_count = 0;
    for (unsigned int j = 0; j < this->classes_count; j++)
        for (unsigned int i = 0; i < this->classes_count; i++)
        {
            this->total_count+= confusion_matrix[j][i];
        }

    this->success_count = 0;
    for (unsigned int j = 0; j < this->classes_count; j++)
        this->success_count+= confusion_matrix[j][j];

    this->miss_count = this->total_count - this->success_count;

    this->accuracy = this->success_count*100.0/(this->total_count + eps);



    for (unsigned int j = 0; j < this->classes_count; j++)
    {
        unsigned int class_sum = 0;
        for (unsigned int i = 0; i < this->classes_count; i++)
        {
            class_sum+= this->confusion_matrix[i][j];
        }

        float class_success_ = this->confusion_matrix[j][j]*100.0/(class_sum + eps);

        this->class_success[j] = class_success_;
    }
}

bool ClassificationCompare::is_nan_error()
{
    return this->nan_error;
}

std::string ClassificationCompare::asString()
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
        result << std::right << std::setw(9) << std::setprecision(5) << class_success[j];

    result << std::right << std::setw(9) <<  "\n\n\n";

    result << "success count" << std::right << std::setw(9) <<  success_count << "\n";
    result << "miss count   " << std::right << std::setw(9) <<  miss_count << "\n";
    result << "accuracy     " << std::right << std::setw(9) <<  accuracy << "%" << "\n";
    result << std::right << std::setw(9) <<   "\n";

    return result.str();
}

Json::Value ClassificationCompare::asJson()
{
    Json::Value result;

    for (unsigned int j = 0; j < confusion_matrix.size(); j++)
      for (unsigned int i = 0; i < confusion_matrix.size(); i++)
        result["confusion_matrix"][j][i] = confusion_matrix[j][i];

    for (unsigned int j = 0; j < confusion_matrix.size(); j++)
      result["class_success_rate"][j] = class_success[j];

    result["success_count"] = success_count;
    result["miss_count"]    = miss_count;
    result["accuracy"]      = accuracy;

    return result;
}

float ClassificationCompare::get_accuracy()
{
    return this->accuracy;
}

std::vector<std::vector<unsigned int>> ClassificationCompare::get_confusion_matrix()
{
    return this->confusion_matrix;
}

unsigned int ClassificationCompare::class_idx(std::vector<float> &v)
{
    unsigned int result = 0;

    for (unsigned int i = 0; i < v.size(); i++)
        if (v[i] > v[result])
            result = i;

    return result;
}


bool ClassificationCompare::is_valid(std::vector<float> &v)
{
    for (unsigned int i = 0; i < v.size(); i++)
        if (isnan(v[i]))
            return false;

    for (unsigned int i = 0; i < v.size(); i++)
        if (isinf(v[i]))
            return false;

    return true;
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
