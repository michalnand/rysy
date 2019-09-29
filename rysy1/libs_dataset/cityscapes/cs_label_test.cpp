#include "cs_label_test.h"
#include <image_save.h>

CSLabelTest::CSLabelTest(std::string polygons_file_name, std::string output_file_name)
{
  this->output_file_name = output_file_name;
  labels.load(polygons_file_name);

  cs_to_class_id.load("input/label_names.json");
}

CSLabelTest::~CSLabelTest()
{

}


void CSLabelTest::process()
{
  /*
  ImageSave image(labels.width(), labels.height());

  auto palette = make_color_palette(cs_to_class_id.get_count());

  unsigned int square_size = 8;

  for (unsigned int y = 0; y < labels.height() - square_size; y+= square_size)
    for (unsigned int x = 0; x < labels.width() - square_size; x+= square_size)
    {
      auto class_name = labels.get_label(x, y, 0);
      int class_id = cs_to_class_id.get(class_name);


    //  class_id = rand()%palette.size();
      if (class_id != -1)
      {
        float r = palette[class_id].x()/256.0;
        float g = palette[class_id].y()/256.0;
        float b = palette[class_id].z()/256.0;

        for (unsigned int ky = 0; ky < square_size; ky++)
        for (unsigned int kx = 0; kx < square_size; kx++)
        {
          image.pixels[y + ky][x + kx].b[0] = r;
          image.pixels[y + ky][x + kx].b[1] = g;
          image.pixels[y + ky][x + kx].b[2] = b;


        }
      }
    }


  image.save(output_file_name, v);
  */
}



std::vector<Point> CSLabelTest::make_color_palette(unsigned int count)
{
  std::vector<Point> result(count);


  unsigned int r = (0*count)/3;
  unsigned int g = (1*count)/3;
  unsigned int b = (2*count)/3;

  for (unsigned int j = 0; j < count; j++)
  {
    result[j].set(r*256.0/count, g*256.0/count, b*256.0/count);

    r = (r + 1)%count;
    g = (g + 1)%count;
    b = (b + 1)%count;
  }


  return result;
}
