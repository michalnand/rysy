#include "polygon.h"
#include <iostream>

Polygon::Polygon()
{

}

Polygon::Polygon(std::vector<Point> &points)
{
  this->points = points;
}

Polygon::Polygon(const std::vector<Point> &points)
{
  this->points = points;
}

Polygon::Polygon(Json::Value &json_polygon)
{
  for (unsigned int i = 0; i < json_polygon.size(); i++)
  {
    Point tmp;

    if (json_polygon[i].size() > 0)
      tmp.set_x(json_polygon[i][0].asInt());

    if (json_polygon[i].size() > 1)
      tmp.set_y(json_polygon[i][1].asInt());

    if (json_polygon[i].size() > 2)
      tmp.set_z(json_polygon[i][2].asInt());


    points.push_back(tmp);
  }
}

Polygon::Polygon(const Json::Value &json_polygon)
{
  for (unsigned int i = 0; i < json_polygon.size(); i++)
  {
    Point tmp;

    if (json_polygon[i].size() > 0)
      tmp.set_x(json_polygon[i][0].asFloat());

    if (json_polygon[i].size() > 1)
      tmp.set_y(json_polygon[i][1].asFloat());

    if (json_polygon[i].size() > 2)
      tmp.set_z(json_polygon[i][2].asFloat());

    points.push_back(tmp);
  }
}

Polygon::Polygon(Polygon& other)
{
  copy(other);
}

Polygon::Polygon(const Polygon& other)
{
  copy(other);
}

Polygon::~Polygon()
{

}

Polygon& Polygon::operator= (Polygon& other)
{
  copy(other);
  return *this;
}

Polygon& Polygon::operator= (const Polygon& other)
{
  copy(other);
  return *this;
}

void Polygon::copy(Polygon& other)
{
  points = other.points;
}

void Polygon::copy(const Polygon& other)
{
  points = other.points;
}



unsigned int Polygon::size()
{
  return points.size();
}

Point Polygon::get_point(unsigned int idx)
{
  return points[idx];
}

void Polygon::set_point(unsigned int idx, Point &point)
{
  points[idx] = point;
}

void Polygon::add_point(Point &point)
{
  points.push_back(point);
}

void Polygon::clear()
{
  points.clear();
}

void Polygon::print()
{
  for (unsigned int i = 0; i < points.size(); i++)
  {
    points[i].print();
    std::cout << "\n";
  }
  std::cout << "\n";
}

bool Polygon::is_point_in(Point &point)
{
  bool result = false;

  int i, j;
  for (i = 0, j = size()-1; i < (int)size(); j = i++)
  {
    if ( ((points[i].y()>point.y()) != (points[j].y()>point.y())) &&
	       (point.x() < (points[j].x()-points[i].x()) * (point.y()-points[i].y()) / (points[j].y()-points[i].y()) + points[i].x())
       )
       result = !result;
  }

  return result;
}
