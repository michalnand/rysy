#ifndef _CS_LABEL_TEST_H_
#define _CS_LABEL_TEST_H_

#include "cs_labels.h"
#include "cs_to_class_id.h"


class CSLabelTest
{
  private:
    std::string output_file_name;
    CSLabels labels;
    CSToClassId cs_to_class_id;

  public:
    CSLabelTest(std::string polygons_file_name, std::string output_file_name);
    virtual ~CSLabelTest();

    void process();

  private:
    std::vector<Point> make_color_palette(unsigned int count);

};

#endif
