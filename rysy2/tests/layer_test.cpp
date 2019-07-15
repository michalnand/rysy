#include <iostream>

#include <layers/layer.h>
#include <layers/fc_layer.h>
#include <layers/dropout_layer.h>

#include <layers/convolution_layer.h>
#include <layers/activation_elu_layer.h>

#include <layers/activation_relu_layer.h>
#include <layers/convolution_layer.h>
#include <layers/max_pooling_layer.h>


void relu_test()
{
    Shape input_shape(5, 3, 2);

    Json::Value parameters;

    parameters["hyperparameters"]["learning_rate"]  = 0.0001;
    parameters["hyperparameters"]["lambda1"]        = 0.0000001;
    parameters["hyperparameters"]["lambda2"]        = 0.0000001;
    parameters["hyperparameters"]["dropout"]        = 0.5;

    Tensor input(input_shape);
    Tensor output(input_shape);

    ActivationReluLayer layer(input_shape, parameters);
    std::cout << layer.asString() << "\n";

    input.set_random(1.0);
    output.clear();

    layer.forward(output, input);

    std::cout << "INPUT : \n";
    input.print();

    std::cout << "OUTPUT : \n";
    output.print();
}


void elu_test()
{
    Shape input_shape(5, 3, 2);

    Json::Value parameters;

    parameters["hyperparameters"]["learning_rate"]  = 0.0001;
    parameters["hyperparameters"]["lambda1"]        = 0.0000001;
    parameters["hyperparameters"]["lambda2"]        = 0.0000001;
    parameters["hyperparameters"]["dropout"]        = 0.5;

    Tensor input(input_shape);
    Tensor output(input_shape);

    ActivationEluLayer layer(input_shape, parameters);
    std::cout << layer.asString() << "\n";

    input.set_random(1.0);
    output.clear();

    layer.forward(output, input);

    std::cout << "INPUT : \n";
    input.print();

    std::cout << "OUTPUT : \n";
    output.print();
}




void dropout_test()
{
    Shape input_shape(32, 32, 3);

    Json::Value parameters;

    parameters["hyperparameters"]["learning_rate"]  = 0.0001;
    parameters["hyperparameters"]["lambda1"]        = 0.0000001;
    parameters["hyperparameters"]["lambda2"]        = 0.0000001;
    parameters["hyperparameters"]["dropout"]        = 0.5;

    Tensor input(input_shape);
    Tensor output(input_shape);

    DropoutLayer layer(input_shape, parameters);
    std::cout << layer.asString() << "\n";

    //input.set_random(1.0);
    input.set_const(1.0);
    output.clear();

    layer.set_training_mode();

    layer.forward(output, input);

    std::cout << "INPUT : \n";
    input.print();

    std::cout << "OUTPUT : \n";
    output.print();
}

void convolution_test()
{
    Shape input_shape(8, 8, 24);

    Json::Value parameters;

    parameters["hyperparameters"]["learning_rate"]  = 0.0001;
    parameters["hyperparameters"]["lambda1"]        = 0.0000001;
    parameters["hyperparameters"]["lambda2"]        = 0.0000001;
    parameters["hyperparameters"]["dropout"]        = 0.5;

    parameters["shape"][0] = 3;
    parameters["shape"][1] = 3;
    parameters["shape"][2] = 64;


    Tensor input(input_shape);

    ConvolutionLayer layer(input_shape, parameters);
    std::cout << layer.asString() << "\n";

    Tensor output(layer.get_output_shape());


    input.set_random(1.0);
    //input.set_const(1.0);
    output.clear();

    layer.set_training_mode();

    layer.forward(output, input);

    std::cout << "INPUT : \n";
    input.print();

    std::cout << "OUTPUT : \n";
    output.print();
}




void pooling_test()
{
    Shape input_shape(8, 8, 3);

    Json::Value parameters;

    parameters["hyperparameters"]["learning_rate"]  = 0.0001;
    parameters["hyperparameters"]["lambda1"]        = 0.0000001;
    parameters["hyperparameters"]["lambda2"]        = 0.0000001;
    parameters["hyperparameters"]["dropout"]        = 0.5;

    parameters["shape"][0] = 2;
    parameters["shape"][1] = 2;

    Tensor input(input_shape);

    MaxPoolingLayer layer(input_shape, parameters);
    std::cout << layer.asString() << "\n";

    Tensor output(layer.get_output_shape());

    input.set_random(1.0);
    //input.set_const(1.0);
    output.clear();

    layer.set_training_mode();

    layer.forward(output, input);

    std::cout << "INPUT : \n";
    input.print();

    std::cout << "OUTPUT : \n";
    output.print();
}

int main()
{
    srand(time(NULL));

    //relu_test();
    //elu_test();
    //dropout_test();
    convolution_test();
    //pooling_test();


    std::cout << "program done\n";
    return 0;
}
