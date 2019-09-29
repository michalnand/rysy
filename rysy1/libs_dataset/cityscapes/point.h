#ifndef _POINT_H_
#define _POINT_H_

typedef float point_t;

#include <vector>

class Point
{
  private:
    point_t m_x, m_y, m_z;

  public:
    // Default constructor
    Point(point_t x = 0, point_t y = 0, point_t z = 0);

    Point(std::vector<point_t> &v);
    Point(const std::vector<point_t> &v);


    // Copy constructor
    Point(Point& other);

    // Copy constructor
    Point(const Point& other);

    // Destructor
    virtual ~Point();

    // Copy assignment operator
    Point& operator= (Point& other);

    // Copy assignment operator
    Point& operator= (const Point& other);

  protected:
    void copy(Point& other);
    void copy(const Point& other);

  public:
    point_t x();
    point_t y();
    point_t z();

    void set_x(point_t x);
    void set_y(point_t y);
    void set_z(point_t z);

    void set(point_t x, point_t y = 0, point_t z = 0);


  public:
    void print();

  public:
    point_t length();


    Point& operator+ (Point& other);
    Point& operator+ (const Point& other);

    Point& operator- (Point& other);
    Point& operator- (const Point& other);

    point_t operator* (Point& other);
    point_t operator* (const Point& other);
};


#endif
