#ifndef _EXPORT_STL10_H_
#define _EXPORT_STL10_H_

#include <string>
#include <vector>


class ExportSTL10
{
    public:
        ExportSTL10(std::string input_file_name, std::string output_dir);
        virtual ~ExportSTL10();

    public:
        void process(unsigned int max_count);

    private:
        std::string input_file_name;
        std::string output_dir;
};


#endif
