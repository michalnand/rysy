#ifndef _POLYGON_H_
#define _POLYGON_H_

#include "point.h"

#include <json_config.h>

class Polygon
{
  protected:
    std::vector<Point> points;

  public:
    Polygon();
    Polygon(std::vector<Point> &points);
    Polygon(const std::vector<Point> &points);

    Polygon(Json::Value &json_polygon);
    Polygon(const Json::Value &json_polygon);

    // Copy constructor
    Polygon(Polygon& other);

    // Copy constructor
    Polygon(const Polygon& other);

    // Destructor
    virtual ~Polygon();

    // Copy assignment operator
    Polygon& operator= (Polygon& other);

    // Copy assignment operator
    Polygon& operator= (const Polygon& other);

  protected:
    void copy(Polygon& other);
    void copy(const Polygon& other);

  public:
    unsigned int size();

    Point get_point(unsigned int idx);
    void set_point(unsigned int idx, Point &point);
    void add_point(Point &point);
    void clear();

    void print();

    bool is_point_in(Point &point);
};

#endif
