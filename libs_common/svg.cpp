#include "svg.h"

SVG::SVG()
{

}

SVG::~SVG()
{

}

void SVG::save(std::string file_name)
{
  FILE *f;
  f = fopen(file_name.c_str(), "w");
  fprintf(f,"%s", svg.c_str());
  fclose(f);
}


void SVG::add_header(unsigned int width, unsigned int height)
{
  svg = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
  svg+= "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" ";
  svg+= "width=\"" + std::to_string(width) + "\" ";
  svg+= "height=\"" + std::to_string(height) + "\" ";
  svg+= ">\n";
}

void SVG::add_footer()
{
  svg+= "</svg>";
}

void SVG::add_rectangle(
                                  int x,
                                  int y,
                                  int width,
                                  int height,

                                  int swidth,

                                  int r, int g, int b,
                                  int sr, int sg, int sb
                                 )
{
  svg+= "<rect ";
  svg+= "x=\"" + std::to_string(x) + "\" ";
  svg+= "y=\"" + std::to_string(y) + "\" ";

  svg+= "width=\""  + std::to_string(width) + "\" ";
  svg+= "height=\"" + std::to_string(height) + "\" ";

  svg+= "fill=\"rgb(";
  svg+= std::to_string(r)+","+std::to_string(g)+","+std::to_string(b)+")\" ";

  svg+= "stroke-width=\"" + std::to_string(swidth) + "\" ";

  svg+= "stroke=\"rgb(";
  svg+= std::to_string(sr)+","+std::to_string(sg)+","+std::to_string(sb)+")\" ";

  svg+= "/>\n";
}

void SVG::add_circle(  int x,
                       int y,
                       int radius,

                       int swidth,

                       int r, int g, int b,
                       int sr, int sg, int sb)
{
  svg+= "<circle ";
  svg+= "cx=\"" + std::to_string(x) + "\" ";
  svg+= "cy=\"" + std::to_string(y) + "\" ";
  svg+= "r=\""  + std::to_string(radius) + "\" ";

  svg+= "fill=\"rgb(";
  svg+= std::to_string(r)+","+std::to_string(g)+","+std::to_string(b)+")\" ";

  svg+= "stroke-width=\"" + std::to_string(swidth) + "\" ";

  svg+= "stroke=\"rgb(";
  svg+= std::to_string(sr)+","+std::to_string(sg)+","+std::to_string(sb)+")\" ";

  svg+= "/>\n";
}


void SVG::add_line( int x,
                    int y,

                    int x_end,
                    int y_end,

                    int swidth,

                    int r, int g, int b,
                    int sr, int sg, int sb)
{
   svg+= "<line ";
   svg+= "x1=\"" + std::to_string(x) + "\" ";
   svg+= "y1=\"" + std::to_string(y) + "\" ";

   svg+= "x2=\"" + std::to_string(x_end) + "\" ";
   svg+= "y2=\"" + std::to_string(y_end) + "\" ";

   svg+= "fill=\"rgb(";
   svg+= std::to_string(r)+","+std::to_string(g)+","+std::to_string(b)+")\" ";

   svg+= "stroke-width=\"" + std::to_string(swidth) + "\" ";

   svg+= "stroke=\"rgb(";
   svg+= std::to_string(sr)+","+std::to_string(sg)+","+std::to_string(sb)+")\" ";

   svg+= "/>\n";
}

void SVG::add_text( int x, int y,
                              std::string text,
                              int size,
                              bool vertical
                            )
{
  svg+= "<text ";
  svg+= "x=\"" + std::to_string(x) + "\" ";
  svg+= "y=\"" + std::to_string(y) + "\" ";

  svg+= "style=\"";
  svg+= "font-size: "+std::to_string(size)+"px; ";

  if (vertical)
    svg+= "writing-mode: tb; ";

  svg+= "\">";

  svg+= text;

  svg+= "</text>\n";
}
