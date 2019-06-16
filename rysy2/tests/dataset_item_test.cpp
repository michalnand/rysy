#include <iostream>
#include <dataset_item.h>



int main()
{
    srand(time(NULL));

    Shape input_shape(8, 8, 3), output_shape(1, 1, 10);

    DatasetItem item(input_shape, output_shape, 8);


    item._set_random();

    item.print(true);

    std::cout << "program done\n";
    return 0;
}
