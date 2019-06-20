using System;

namespace ConsoleApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            String dataset_path = "/home/michal/dataset/mnist/";

            //load dataset
            DatasetMnist dataset = new DatasetMnist(dataset_path + "train-images.idx3-ubyte",
                                                    dataset_path + "train-labels.idx1-ubyte",
                                                    dataset_path + "t10k-images.idx3-ubyte",
                                                    dataset_path + "t10k-labels.idx1-ubyte");

            /*
            create example network
            3 convolutional layers, dropout and full connected layer as output
            C3x3x16 - P2x2 - C3x3x32 - P2x2 - C3x3x32 - FC10
            */

            CNN cnn = new CNN(dataset.get_input_shape(), dataset.get_output_shape(), (float)0.0025);

            cnn.add_layer("convolution", new Shape(3, 3, 16));
            cnn.add_layer("elu");
            cnn.add_layer("max_pooling", new Shape(2, 2));
            cnn.add_layer("convolution", new Shape(3, 3, 32));
            cnn.add_layer("elu");
            cnn.add_layer("max_pooling", new Shape(2, 2));
            cnn.add_layer("convolution", new Shape(3, 3, 32));
            cnn.add_layer("elu");
            cnn.add_layer("dropout");
            cnn.add_layer("output");

            cnn.print();

            //train network - set epoch count
            uint epoch_count = 1;
            cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), epoch_count);


            // test network response on whole testing dataset items

            ClassificationCompare compare = new ClassificationCompare(dataset.get_classes_count());

            //neural network output - vector of floats
            SWIGTYPE_p_std__vectorT_float_t nn_output = rysy.VectorFloatCreate(dataset.get_classes_count());

            //for all testing items
            for (uint item_idx = 0; item_idx < dataset.get_testing_count(); item_idx++)
            {
                //get network response
                cnn.forward(nn_output, dataset.get_testing_input(item_idx));

                //compare with testing dataset
                compare.add(dataset.get_testing_output(item_idx), nn_output);

                if (compare.is_nan_error())
                    Console.WriteLine("NaN error");
            }

            //process computing and print results
            compare.compute();
            Console.WriteLine(compare.asString());

            Console.WriteLine("program done");
        }
    }
}
