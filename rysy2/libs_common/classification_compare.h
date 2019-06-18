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

        ClassificationCompare(unsigned int classes_count);

        virtual ~ClassificationCompare();

        ClassificationCompare& operator= (ClassificationCompare& other);
        ClassificationCompare& operator= (const ClassificationCompare& other);

    protected:
        void copy(ClassificationCompare& other);
        void copy(const ClassificationCompare& other);


    public:
        void init(unsigned int classes_count);
        int add(std::vector<float> &target_output, std::vector<float> &predicted_output);
        void compute();

        bool is_nan_error();

    public:
        std::string asString();
        Json::Value asJson();

    private:
        unsigned int class_idx(std::vector<float> &v);
        bool is_valid(std::vector<float> &v);


    private:
        unsigned int classes_count;

        std::vector<std::vector<unsigned int>> confusion_matrix;

        std::vector<float> class_success;

        unsigned int success_count, miss_count, total_count;
        float accuracy;

        bool nan_error;
};

#endif
