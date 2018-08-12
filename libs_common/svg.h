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

    void add_rectangle(
                                      int x,
                                      int y,
                                      int width,
                                      int height,

                                      int swidth,

                                      int r, int g, int b,
                                      int sr, int sg, int sb
                                    );

    void add_text( int x, int y,
                   std::string text,
                   int size,
                   bool vertical
                 );
};

#endif
