#include <iostream>
#include <dataset_mnist.h>
#include <cnn.h>

#include <classification_compare.h>
#include <layers/fc_layer.h>

int main()
{
    std::string dataset_path = "/home/michal/dataset/mnist/";

    DatasetMnist dataset(   dataset_path + "train-images.idx3-ubyte",
                            dataset_path + "train-labels.idx1-ubyte",
                            dataset_path + "t10k-images.idx3-ubyte",
                            dataset_path + "t10k-labels.idx1-ubyte");


    std::cout << "\n\n\n\n";

    CNN cnn(dataset.get_input_shape(), dataset.get_output_shape(), 0.001);

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
    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");
    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");
    cnn.add_layer("dropout");
    cnn.add_layer("output");

    //print network info
    std::cout << cnn.asString() << "\n";

    //start training
    std::cout << "training\n";
    cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), 5);

    cnn.save("mnist_0/");


    /*
    CNN cnn(std::string("mnist_0/network_config.json"));
    std::cout << cnn.asString() << "\n";
    */

    std::cout << "testing\n";
    ClassificationCompare compare(dataset.get_classes_count());

    std::vector<float> nn_output(dataset.get_classes_count());
    for (unsigned int item_idx = 0; item_idx < dataset.get_testing_count(); item_idx++)
    {
        cnn.forward(nn_output, dataset.get_testing_input(item_idx));
        compare.add(dataset.get_testing_output(item_idx), nn_output);

        if (compare.is_nan_error())
        {
            std::cout << "NaN error\n";
            break;
        }
    }

    compare.compute();

    std::cout << compare.asString() << "\n";


    std::cout << "program done\n";
    return 0;
}
