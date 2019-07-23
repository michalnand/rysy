#include <iostream>
#include <kernels/convolution_layer_forward.cuh>
#include <kernels/convolution_layer_backward.cuh>


int forward_test(unsigned int width, unsigned int height, unsigned int channels, unsigned int kernel_size)
{
    float value = ((rand()%10000)/10000.0) + 0.1;

    unsigned int kernel_size_half = kernel_size/2;

    Shape kernel_shape(kernel_size, kernel_size, 1);

    Shape input_shape(width, height, channels);

    Tensor kernel(kernel_shape.w(), kernel_shape.h(), kernel_shape.d()*channels);
    kernel.set_const(1.0/(kernel_shape.w()*kernel_shape.h()*channels));

    Tensor bias(1, 1, 1);
    bias.clear();

    Tensor input(input_shape);
    input.set_const(value);

    Tensor output(input_shape.w(), input_shape.h(), 1);
    output.clear();

    Tensor target_output(input_shape.w(), input_shape.h(), kernel_shape.d());
    target_output.clear();
    for (unsigned int y = kernel_size_half; y < target_output.h()-kernel_size_half; y++)
        for (unsigned int x = kernel_size_half; x < target_output.w()-kernel_size_half; x++)
            target_output.set(x, y, 0, 0, value);



    convolution_layer_forward(output, input, kernel, bias);


    target_output.sub(output);

    float error_sum = 0.0;
    for (unsigned int y = 0; y < target_output.h(); y++)
        for (unsigned int x = 0; x < target_output.w(); x++)
        {
            float tmp = target_output.get(x, y, 0, 0);
            error_sum+= tmp*tmp;
        }

    if (error_sum > 0.001)
    {
        std::cout << "FORWARD test ERROR\n";
        std::cout << "error_value = " << error_sum << "\n";

        std::cout << "width = " << width << "\n";
        std::cout << "height = " << height << "\n";
        std::cout << "channels = " << channels << "\n";
        std::cout << "kernel_size = " << kernel_size << "\n";

        /*
        std::cout << "input = " << "\n";
        input.print();

        std::cout << "output = " << "\n";
        output.print();
        */
        std::cout << "error = " << "\n";
        target_output.print();

        return -1;
    }

    return 0;
}



int backward_test(unsigned int width, unsigned int height, unsigned int channels, unsigned int kernel_size)
{
    Shape kernel_shape(kernel_size, kernel_size, 1);

    Tensor kernel(kernel_shape.w(), kernel_shape.w(), kernel_shape.d()*channels);
    kernel.set_const(1.0/(kernel_shape.w()*kernel_shape.w()*kernel_shape.d()*channels));

    unsigned int kernel_size_half = kernel_size/2;

    float error_value   = 1; //0.1 + (rand()%10000)/10000.0;
    float target_value  = error_value;

    Tensor error(width, height, 1);
    error.clear();
    for (unsigned int y = kernel_size_half; y < error.h()-kernel_size_half; y++)
        for (unsigned int x = kernel_size_half; x < error.w()-kernel_size_half; x++)
            error.set(x, y, 0, 0, error_value);


    Tensor error_back(width, height, channels);
    error_back.clear();

    convolution_layer_backward(error_back, error, kernel);

    std::cout << "error = \n";
    error.print();

    std::cout << "error_back = \n";
    error_back.print();

    std::cout << "target_value = " << target_value << "\n";

    /*
    float value = ((rand()%10000)/10000.0) + 0.1;

    unsigned int kernel_size_half = kernel_size/2;


    Shape kernel_shape(kernel_size, kernel_size, 1);
    Shape input_shape(width, height, channels);

    Tensor kernel(kernel_shape.w(), kernel_shape.h(), kernel_shape.d()*input_shape.d());
    kernel.set_const(1.0/(kernel_shape.w()*kernel_shape.h()*channels));

    Tensor bias(1, 1, kernel_shape.d());
    bias.clear();

    Tensor input(input_shape);
    input.set_const(value);

    Tensor output(input_shape.w(), input_shape.h(), kernel_shape.d());
    output.clear();

    Tensor target_output(input_shape.w(), input_shape.h(), kernel_shape.d());
    target_output.clear();
    for (unsigned int y = kernel_size_half; y < target_output.h()-kernel_size_half; y++)
        for (unsigned int x = kernel_size_half; x < target_output.w()-kernel_size_half; x++)
            target_output.set(x, y, 0, 0, value);



    convolution_layer_forward(output, input, kernel, bias);


    target_output.sub(output);

    float error_sum = 0.0;
    for (unsigned int y = 0; y < target_output.h(); y++)
        for (unsigned int x = 0; x < target_output.w(); x++)
        {
            float tmp = target_output.get(x, y, 0, 0);
            error_sum+= tmp*tmp;
        }

    if (error_sum > 0.001)
    {
        std::cout << "FORWARD test ERROR\n";
        std::cout << "error_value = " << error_sum << "\n";

        std::cout << "width = " << width << "\n";
        std::cout << "height = " << height << "\n";
        std::cout << "channels = " << channels << "\n";
        std::cout << "kernel_size = " << kernel_size << "\n";


        std::cout << "error = " << "\n";
        target_output.print();

        return -1;
    }
    */
    return 0;
}

int main()
{
    std::vector<unsigned int> width =   {3, 3, 4, 4, 4, 5, 5, 6, 3, 6, 4, 6, 7, 8, 8, 3, 9, 10, 13, 16, 9, 17, 19, 19, 20, 20, 28, 32, 48, 50, 51, 64, 60, 40};
    std::vector<unsigned int> height =  {3, 4, 3, 4, 5, 4, 5, 3, 6, 4, 6, 6, 7, 8, 3, 8, 9, 10, 13, 9, 16, 17, 19, 20, 19, 20, 28, 32, 48, 50, 51, 64, 40, 60};

    std::vector<unsigned int> kernel_size = {1, 3, 5};
    std::vector<unsigned int> channels    = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 19, 20, 32, 48, 64, 96, 100, 128, 256, 512, 1024};

    /*
    for (unsigned int ch = 0; ch < channels.size(); ch++)
        for (unsigned int i = 0; i < width.size(); i++)
            for (unsigned int k = 0; k < kernel_size.size(); k++)
                if (width[i] >= kernel_size[k])
                if (height[i] >= kernel_size[k])
                {
                    int res = forward_test(width[i], height[i], channels[ch], kernel_size[k]);
                    if (res == 0)
                        std::cout << "forward OK for " << width[i] << " " << height[i] << " " << channels[ch] << " " << kernel_size[k] << "\n";
                    else
                        return 0;
                }
    */

    backward_test(8, 8, 1, 3);

    return 0;
}
