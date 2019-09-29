#ifndef _CLASSIFICATION_COMPARE_H_
#define _CLASSIFICATION_COMPARE_H_


#include <string>
#include <vector>

#include "json_config.h"

class ClassificationCompare
{
  private:
    unsigned int classes_count;
    unsigned int top_n_count;

    std::vector<std::vector<unsigned int>> confusion_matrix;
    std::vector<float> class_success_rate;

    unsigned int sum_all, sum_correct;

    float succes_result;

    std::vector<float> summary_result;
    std::string text_result;
    Json::Value json_result;

  public:
    ClassificationCompare();
    ClassificationCompare(unsigned int classes_count_, unsigned int top_n_count_ = 1);

    virtual ~ClassificationCompare();

    void compare(std::vector<float> &required_value, std::vector<float> &output_value);

    void set_classes_count(unsigned int classes_count);
    void set_top_n_count(unsigned int top_n_count_);

    void clear();
    float process(bool percent_result);
    float get_success();

    std::vector<float> get_summary_result()
    {
      return summary_result;
    }

    std::string get_text_result()
    {
      return text_result;
    }

    Json::Value get_json_result()
    {
      return json_result;
    }

    void save_text_file(std::string file_name);
    void save_json_file(std::string file_name);

    std::vector<std::vector<unsigned int>>& get_confusion_matrix()
    {
      return confusion_matrix;
    }


    std::vector<float>& get_class_success_rate()
    {
      return class_success_rate;
    }

    unsigned int get_classes_count()
    {
	return classes_count;
    }
	

  private:

    std::vector<float> process_summary_result();
    std::string process_text_result();
    Json::Value process_json_result();

    unsigned int argmax(std::vector<float> &v, unsigned int size);

  private:

    std::vector<unsigned int> get_top_n(std::vector<float> &confidence, unsigned int top_n_count);
    bool not_in(std::vector<unsigned int> &vect, unsigned int value);

};

#endif
