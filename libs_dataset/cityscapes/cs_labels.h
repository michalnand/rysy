#ifndef _CS_LABELS_H_
#define _CS_LABELS_H_

#include "polygon.h"

#include <string>

class CSLabels
{
  protected:
    int m_width, m_height;
    unsigned int m_objects_count;

    std::vector<std::string>  labels;
    std::vector<Polygon>      polygons;

  public:
    // Default constructor
    CSLabels();

    // Copy constructor
    CSLabels(CSLabels& other);

    // Copy constructor
    CSLabels(const CSLabels& other);

    CSLabels(std::string json_file_name);
    CSLabels(Json::Value json_config);

    // Destructor
    virtual ~CSLabels();

    // Copy assignment operator
    CSLabels& operator= (CSLabels& other);

    // Copy assignment operator
    CSLabels& operator= (const CSLabels& other);

    void load(std::string json_file_name);
    void load(Json::Value &json_config);


    std::string get_label(Point &point);
    std::string get_label(point_t x = 0, point_t y = 0, point_t z = 0);

  public:
    unsigned int objects_count()
    {
      return m_objects_count;
    }

    int width()
    {
      return m_width;
    }

    int height()
    {
      return m_height;
    }

    void print(bool print_polygons = false);

  protected:
    void copy(CSLabels& other);
    void copy(const CSLabels& other);

};

#endif
