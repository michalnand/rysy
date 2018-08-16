#include <iostream>

#include <cityscapes/cs_label_test.h>


int main()
{
  srand(time(NULL));

  CSLabelTest label_test("input/munster_000173_000019_gtCoarse_polygons.json", "input/munster_000173_000019_leftImg8bit_exported.png");
  label_test.process();

  printf("program done\n");

  return 0;
}
