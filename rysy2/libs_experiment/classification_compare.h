#ifndef _CLASSIFICATION_COMPARE_H_
#define _CLASSIFICATION_COMPARE_H_

#include <vector>

#include <string>
#include <jsoncpp/json/value.h>

class ClassificationCompare
{
    public:
        ClassificationCompare();
        ClassificationCompare(ClassificationCompare& other);
        ClassificationCompare(const ClassificationCompare& other);

        ClassificationCompare(unsigned int classes_count, unsigned int top_n_count = 1);

        virtual ~ClassificationCompare();

        ClassificationCompare& operator= (ClassificationCompare& other);
        ClassificationCompare& operator= (const ClassificationCompare& other);

    protected:
        void copy(ClassificationCompare& other);
        void copy(const ClassificationCompare& other);


    public:
        void init(unsigned int classes_count, unsigned int top_n_count = 1);
        int add(std::vector<float> &target_output, std::vector<float> &predicted_output);
        void compute();

        bool is_nan_error();

    public:
        std::string asString();
        Json::Value asJson();

    public:
        float get_accuracy();
        std::vector<std::vector<unsigned int>> get_confusion_matrix();

    private:
        unsigned int class_idx(std::vector<float> &v);
        bool is_valid(std::vector<float> &v);

        std::vector<unsigned int> get_top_n(std::vector<float> &confidence, unsigned int top_n_count);
        bool not_in(std::vector<unsigned int> &vect, unsigned int value);

    private:
        unsigned int classes_count;

        std::vector<std::vector<unsigned int>> confusion_matrix;

        std::vector<float> class_success;

        unsigned int success_count, miss_count, total_count;
        float accuracy;

        bool nan_error;
        unsigned int top_n_count;
};

#endif
