#include <iostream>
#include <dqn.h>



int main()
{
    Shape input_shape(64, 64, 12);
    Shape output_shape(1, 1, 8);

    Tensor input(input_shape), output(output_shape);



    CNN cnn(input_shape, output_shape, 0.001);

    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("elu");
    cnn.add_layer("max pooling", Shape(2, 2, 1));

    cnn.add_layer("convolution", Shape(3, 3, 64));
    cnn.add_layer("elu");
    cnn.add_layer("max pooling", Shape(2, 2, 1));

    cnn.add_layer("convolution", Shape(3, 3, 64));
    cnn.add_layer("elu");
    cnn.add_layer("max pooling", Shape(2, 2, 1));

    cnn.add_layer("convolution", Shape(3, 3, 64));
    cnn.add_layer("elu");
    cnn.add_layer("max pooling", Shape(2, 2, 1));

    cnn.add_layer("fc", Shape(1, 1, 256));
    cnn.add_layer("elu");

    cnn.add_layer("dropout");

    cnn.add_layer("output");

    cnn.print();



    cnn.set_training_mode();
    for (unsigned int i = 0; i < 1000; i++)
    {
        input.set_random(1.0);
        output.set_random(1.0);
        cnn.train(output, input);
    }

    std::cout << "program done\n";
    return 0;
}
