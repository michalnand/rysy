#ifndef _REGRESSION_COMPARE_H_
#define _REGRESSION_COMPARE_H_


#include <string>
#include <vector>

#include <json_config.h>
#include <histogram.h>

class RegressionCompare
{
    public:
        RegressionCompare();
        RegressionCompare(unsigned int output_size);

        virtual ~RegressionCompare();

        void set_output_size(unsigned int output_size);

        int add(std::vector<float> &required_value, std::vector<float> &output_value);
        void compute(int fixed_bars_count = -1);
        bool is_nan_error();

        float get_error_average();
        float get_error_std();

        float get_error_average_euclidean();
        float get_error_std_euclidean();

        void save_text_file(std::string log_file_name_prefix);
        void save_json_file(std::string json_file_name);
        Json::Value process_json_result();

    protected:
        bool is_valid(std::vector<float> &v);

    private:
        unsigned int output_size;

        //full results - separated for each axis
        std::vector<Histogram> h_required;  //target values histogram
        std::vector<Histogram> h_resulted;  //computed values histogram
        std::vector<Histogram> h_error;     //errors histogram

        //full results - common for each axis
        Histogram h_required_summary;
        Histogram h_resulted_summary;
        Histogram h_error_summary;

        //error computed euclidean distance of target and observed values
        Histogram h_error_summary_euclidean;

        unsigned int count;

        Json::Value json_result;

        bool nan_error;
};

#endif
