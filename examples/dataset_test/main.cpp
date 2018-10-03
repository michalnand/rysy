#include <iostream>

#include <trajectory_dataset/dat_load.h>
#include <trajectory_dataset/dats_load.h>
#include <glvisualisation.h>

int main()
{
  DatsLoad dataset("training.json");

  dataset.normalise_column();
  dataset.print();

  GLVisualisation visualisation;

  unsigned int step = 0;

  while (1)
  {
    visualisation.start();

    visualisation.translate(0.0, 0.0, -3.0);

    // visualisation.rotate(-60.0, 0.0, 0.0);

    for (unsigned int cell = 0; cell < dataset.get_dat_count(); cell++)
    {
      float x = 2.0*dataset.get(cell, 1, step) - 1.0;
      float y = 2.0*dataset.get(cell, 2, step) - 1.0;
      float z = 0.1*dataset.get(cell, 3, step);

      visualisation.push();
      visualisation.set_color(1.0, 0.0, 0.0);
      visualisation.translate(x, y, z);
      visualisation.paint_sphere(0.05);
      visualisation.pop();
    }

    visualisation.finish();

    step = (step + 1)%dataset.get_lines_count();
  }

  std::cout << "program done\n";
  return 0;
}
