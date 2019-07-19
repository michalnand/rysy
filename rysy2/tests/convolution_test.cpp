#include <iostream>
#include <kernels/convolution_layer_forward.cuh>
#include <kernels/convolution_layer_backward.cuh>

void forward_test(Shape input_shape, Shape kernel_shape)
{
    Tensor kernel(kernel_shape.w(), kernel_shape.h(), kernel_shape.d()*input_shape.d());
    kernel.set_random(0.1);

    Tensor bias(1, 1, kernel_shape.d());
    bias.clear();

    Tensor input(input_shape);
    input.set_random(1.0);

    Tensor output(input_shape.w(), input_shape.h(), kernel_shape.d());

    std::cout << "kernel = \n";
    kernel.print();

    std::cout << "input = \n";
    input.print();

    convolution_layer_forward(output, input, kernel, bias);


    std::cout << "output = \n";
    output.print();
}

int main()
{
    Shape input_shape(3, 3, 16);
    Shape kernel_shape(3, 3, 64);

    forward_test(input_shape, kernel_shape);

    std::cout << "program done\n";
    return 0;
}
