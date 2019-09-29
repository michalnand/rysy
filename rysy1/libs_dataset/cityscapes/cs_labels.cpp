#include "cs_labels.h"

#include <iostream>

CSLabels::CSLabels()
{
  m_width         = 0;
  m_height        = 0;
  m_objects_count = 0;
}

CSLabels::CSLabels(CSLabels& other)
{
  copy(other);
}

CSLabels::CSLabels(const CSLabels& other)
{
  copy(other);
}

CSLabels::CSLabels(std::string json_file_name)
{
  load(json_file_name);
}

CSLabels::CSLabels(Json::Value json_config)
{
  load(json_config);
}

CSLabels::~CSLabels()
{

}

CSLabels& CSLabels::operator= (CSLabels& other)
{
  copy(other);
  return *this;
}

CSLabels& CSLabels::operator= (const CSLabels& other)
{
  copy(other);
  return *this;
}

void CSLabels::load(std::string json_file_name)
{
  JsonConfig json(json_file_name);
  load(json.result);
}

void CSLabels::load(Json::Value &json_config)
{
  this->m_width         = json_config["imgWidth"].asInt();
  this->m_height        = json_config["imgHeight"].asInt();
  this->m_objects_count = json_config["objects"].size();

  for (unsigned int object = 0; object < m_objects_count; object++)
  {
    labels.push_back(json_config["objects"][object]["label"].asString());
    Polygon polygon(json_config["objects"][object]["polygon"]);
    polygons.push_back(polygon);
  }
}


std::string CSLabels::get_label(Point &point)
{
  std::string result = "null";

  if ( (point.x() < 0) || (point.x() > width()) || (point.y() < 0) || (point.y() > height()) )
    return result;

  for (unsigned int j = 0; j < polygons.size(); j++)
  {
    if (polygons[j].is_point_in(point))
    {
      result = labels[j];
      break;
    }
  }

  return result;
}

std::string CSLabels::get_label(point_t x, point_t y, point_t z)
{
  Point point(x, y, z);
  return get_label(point);
}

void CSLabels::copy(CSLabels& other)
{
  m_width         = other.m_width;
  m_height        = other.m_height;
  m_objects_count = other.m_objects_count;

  labels    = other.labels;
  polygons  = other.polygons;
}

void CSLabels::copy(const CSLabels& other)
{
  m_width         = other.m_width;
  m_height        = other.m_height;
  m_objects_count = other.m_objects_count;

  labels    = other.labels;
  polygons  = other.polygons;
}

void CSLabels::print(bool print_polygons)
{
  for (unsigned int j = 0; j < polygons.size(); j++)
  {
    std::cout << j << " " << labels[j] << "\n";
    if (print_polygons)
    {
      polygons[j].print();
      std::cout << "\n";
    }
  }

  std::cout << "\n";
}
