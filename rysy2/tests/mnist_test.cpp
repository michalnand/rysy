#include <iostream>
#include <dataset_mnist.h>
#include <cnn.h>

#include <classification_compare.h>

#include <dqn.h>
int main()
{
    std::string dataset_path = "/home/michal/dataset/mnist/";

    DatasetMnist dataset(   dataset_path + "train-images.idx3-ubyte",
                            dataset_path + "train-labels.idx1-ubyte",
                            dataset_path + "t10k-images.idx3-ubyte",
                            dataset_path + "t10k-labels.idx1-ubyte");

    CNN cnn(dataset.get_input_shape(), dataset.get_output_shape(), 0.002);


    cnn.add_layer("convolution", Shape(3, 3, 16));
    cnn.add_layer("relu");
    cnn.add_layer("max_pooling", Shape(2, 2));
    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");
    cnn.add_layer("max_pooling", Shape(2, 2));
    cnn.add_layer("convolution", Shape(3, 3, 32));
    cnn.add_layer("relu");
    cnn.add_layer("dropout");
    cnn.add_layer("output");

    //print network info
    cnn.print();

    //start training
    std::cout << "training\n";

    unsigned int epoch_count = 10;

    for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
    {
        cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all());

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
    }


    std::cout << "program done\n";
    return 0;
}
