#include "point.h"

#include <iostream>
#include <math.h>

Point::Point(point_t x, point_t y, point_t z)
{
  m_x = x;
  m_y = y;
  m_z = z;
}

Point::Point(std::vector<point_t> &v)
{
  m_x = 0;
  m_y = 0;
  m_z = 0;

  if (v.size() > 0)
    m_x = v[0];
  if (v.size() > 1)
    m_y = v[1];
  if (v.size() > 2)
    m_z = v[2];
}

Point::Point(const std::vector<point_t> &v)
{
  m_x = 0;
  m_y = 0;
  m_z = 0;

  if (v.size() > 0)
    m_x = v[0];
  if (v.size() > 1)
    m_y = v[1];
  if (v.size() > 2)
    m_z = v[2];
}

Point::Point(Point& other)
{
  copy(other);
}

Point::Point(const Point& other)
{
  copy(other);
}

Point::~Point()
{

}

Point& Point::operator= (Point& other)
{
  copy(other);

  return *this;
}

Point& Point::operator= (const Point& other)
{
  copy(other);

  return *this;
}

void Point::copy(Point& other)
{
  (void)other;
  //TODO copy other to this
}

void Point::copy(const Point& other)
{
  (void)other;
  //TODO copy other to this
}

point_t Point::x()
{
  return m_x;
}

point_t Point::y()
{
  return m_y;
}

point_t Point::z()
{
  return m_z;
}


void Point::set_x(point_t x)
{
  m_x = x;
}

void Point::set_y(point_t y)
{
  m_y = y;
}

void Point::set_z(point_t z)
{
  m_z = z;
}


void Point::print()
{
  std::cout << m_x << " ";
  std::cout << m_y << " ";
  std::cout << m_z << " ";
}


point_t Point::length()
{
  point_t result = 0;

  result+= m_x*m_x;
  result+= m_y*m_y;
  result+= m_z*m_z;

  return sqrt(result);
}

Point& Point::operator+ (Point& other)
{
  m_x+= other.m_x;
  m_y+= other.m_y;
  m_z+= other.m_z;

  return *this;
}

Point& Point::operator+ (const Point& other)
{
  m_x+= other.m_x;
  m_y+= other.m_y;
  m_z+= other.m_z;

  return *this;
}

Point& Point::operator- (Point& other)
{
  m_x-= other.m_x;
  m_y-= other.m_y;
  m_z-= other.m_z;

  return *this;
}

Point& Point::operator- (const Point& other)
{
  m_x-= other.m_x;
  m_y-= other.m_y;
  m_z-= other.m_z;

  return *this;
}

point_t Point::operator* (Point& other)
{
  point_t result = 0;
  return result;
}

point_t Point::operator* (const Point& other)
{
  point_t result = 0;
  result+= m_x*other.m_x;
  result+= m_y*other.m_y;
  result+= m_z*other.m_z;

  return result;
}
