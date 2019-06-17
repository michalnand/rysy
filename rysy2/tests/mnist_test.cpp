#include <iostream>
#include <dataset_mnist.h>
#include <cnn.h>


int main()
{
    std::string dataset_path = "/home/michal/dataset/mnist/";

    DatasetMnist dataset(   dataset_path + "train-images.idx3-ubyte",
                            dataset_path + "train-labels.idx1-ubyte",
                            dataset_path + "t10k-images.idx3-ubyte",
                            dataset_path + "t10k-labels.idx1-ubyte");


    std::cout << "\n\n\n\n";


    CNN cnn(dataset.get_input_shape(), dataset.get_output_shape());


    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");

    cnn.add_layer("max_pooling", Shape(2, 2));

    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");

    cnn.add_layer("max_pooling", Shape(2, 2));

    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");

    cnn.add_layer("dropout");

    //cnn.add_layer("average_pooling", Shape(7, 7));

    cnn.add_layer("output");


    //classification(cnn, dataset, params);
    cnn.train(dataset.get_training_input_all(), dataset.get_training_output_all());

    std::cout << cnn.asString() << "\n";
    std::cout << "program done\n";

    return 0;
}
