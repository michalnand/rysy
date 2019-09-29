#ifndef _CS_PARSE_FILE_H_
#define _CS_PARSE_FILE_H_

#include "cs_labels.h"
#include "cs_to_class_id.h"


class CSParseFile
{
  private:
    std::string input_file_name_prefix;
    std::string input_labels_file_name_prefix;
    std::string ouput_images_dir;
    std::string ouput_images_prefix;
    unsigned int output_image_size;

  private:
    CSLabels labels;
    CSToClassId cs_to_class_id;

  public:
    CSParseFile(  std::string input_file_name_prefix,
                  std::string input_labels_file_name_prefix,
                  std::string ouput_images_dir,
                  std::string ouput_images_prefix,
                  unsigned int output_image_size = 32);

    virtual ~CSParseFile();

    void process_all(unsigned int image_step);
    void process_random(unsigned int count);
/*
  private:
    std::vector<Point> make_color_palette(unsigned int count);
*/
};

#endif
