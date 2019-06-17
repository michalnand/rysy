#include <iostream>
#include <cnn.h>


int main()
{
    Shape input_shape(28, 28, 1);
    Shape output_shape(1, 1, 10);

    CNN cnn(input_shape, output_shape);

    cnn.add_layer("convolution", Shape(3, 3, 16));
    cnn.add_layer("relu");
    cnn.add_layer("convolution", Shape(3, 3, 16));
    cnn.add_layer("relu");

    cnn.add_layer("max_pooling", Shape(2, 2));

    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");
    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");

    cnn.add_layer("max_pooling", Shape(2, 2));

    cnn.add_layer("output");


    std::cout << cnn.asString() << "\n";

    std::cout << "program done\n";
    return 0;
}
