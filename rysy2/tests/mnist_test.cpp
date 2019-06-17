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


    std::cout << cnn.asString() << "\n";


    //classification(cnn, dataset, params);
    cnn.train(dataset.get_training_input_all(), dataset.get_training_output_all());

    unsigned int output_size = dataset.get_output_shape().size();
    std::vector<float> nn_output(output_size);
    for (unsigned int item_idx = 0; item_idx < dataset.get_testing_count(); item_idx++)
    {
        cnn.forward(nn_output, dataset.get_testing_input(item_idx));

        if ((item_idx%10) == 0)
        {
            for (unsigned int i = 0; i < output_size; i++)
                std::cout << dataset.get_testing_output(item_idx)[i] << " ";
            std::cout << "\n";

            for (unsigned int i = 0; i < output_size; i++)
                std::cout << nn_output[i] << " ";

            std::cout << "\n\n";
        }
    }


    std::cout << "program done\n";
    return 0;
}
