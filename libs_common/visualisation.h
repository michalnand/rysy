#ifndef _VISUALISATION_H_
#define _VISUALISATION_H_

#include <vector>
#include <string>

class Visualisation
{
  private:
    float time_;

  public:
    Visualisation(unsigned int window_width = 800, unsigned int window_height = 600);
    virtual ~Visualisation();

    void start();
    void finish();

    void push();
    void pop();

    void translate(float x, float y, float z);
    void rotate(float angle_x, float angle_y, float angle_z);
    void set_color(float r, float g, float b);

    void paint_square(float size);
    void paint_rectangle(float width, float height);

    void paint_line(float x0, float y0, float z0,
                    float x1, float y1, float z1);


    void draw_cube(float size);

    void paint_circle(float size, unsigned int steps = 12);
    void print(float x, float y, float z, std::string string, bool small_font = false);

};


#endif
