using System;

namespace ConsoleApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            string dataset_path = "/home/michal/dataset/mnist/";
            Console.WriteLine(dataset_path);

            Shape shape = new Shape(1, 2, 3);
            shape.set(1, 2, 3);
            shape.print();
            /*
            DatasetMnist dataset(dataset_path + "train-images.idx3-ubyte",
                                 dataset_path + "train-labels.idx1-ubyte",
                                 dataset_path + "t10k-images.idx3-ubyte",
                                dataset_path + "t10k-labels.idx1-ubyte");
                                */
        }
    }
} 
