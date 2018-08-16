#include <iostream>

#include <cityscapes/cs_label_test.h>
#include <cityscapes/cs_parse_file.h>


int main()
{
  srand(time(NULL));

  unsigned int output_image_size = 32;
  unsigned int image_step = 16;

  unsigned int images_count = 100;

  {
    CSParseFile file_parse( "input/munster_000171_000019_",
                            "testing/",
                            "a_",
                            output_image_size);

    // file_parse.process_all(image_step);
    file_parse.process_random(images_count);
  }

  {
    CSParseFile file_parse( "input/munster_000172_000019_",
                            "testing/",
                            "b_",
                            output_image_size);

    // file_parse.process_all(image_step);
    file_parse.process_random(images_count);
  }

  {
    CSParseFile file_parse( "input/munster_000173_000019_",
                            "testing/",
                            "c_",
                            output_image_size);

    // file_parse.process_all(image_step);
    file_parse.process_random(images_count);
  }


  printf("program done\n");

  return 0;
}
