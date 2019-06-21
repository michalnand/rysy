#ifndef _SVG_H_
#define _SVG_H_

#include <string>

class SVG
{
  protected:
    std::string svg;

  public:
    SVG();
    ~SVG();

    void save(std::string file_name);

    void add_header(unsigned int width, unsigned int height);
    void add_footer();

    void add_rectangle( int x,
                        int y,
                        int width,
                        int height,

                        int swidth,

                        int r, int g, int b,
                        int sr, int sg, int sb);

    void add_circle( int x,
                     int y,
                     int radius,

                     int swidth,

                     int r, int g, int b,
                     int sr, int sg, int sb);

    void add_line( int x,
                   int y,

                   int x_end,
                   int y_end,

                   int swidth,

                   int r, int g, int b,
                   int sr, int sg, int sb);

    void add_text( int x, int y,
                   std::string text,
                   int size,
                   bool vertical
                 );

    void add_image_link(int x, int y, int width, int height, std::string file_name);
};

#endif
