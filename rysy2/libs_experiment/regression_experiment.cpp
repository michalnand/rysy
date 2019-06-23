#include <regression_experiment.h>
#include <iostream>

#include <batch.h>
#include <cnn.h>
#include <regression_compare.h>

#include <vector_to_image.h>

RegressionExperiment::RegressionExperiment(DatasetInterface &dataset, std::string experiment_dir, std::string network_config_file)
{
    this->dataset                   = &dataset;
    this->experiment_dir            = experiment_dir;
    this->network_config_file       = experiment_dir + network_config_file;
}



RegressionExperiment::~RegressionExperiment()
{

}

void RegressionExperiment::run()
{
    std::string s_delimiter = delimiter();

    JsonConfig json_config(experiment_dir + "experiment_config.json");

    Log experiment_log(experiment_dir + "experiment.log");
    Log progress_log(experiment_dir + "progress.log");


    unsigned int epoch_count            = json_config.result["epoch_count"].asInt();
    unsigned int training_items_count   = dataset->get_training_count();
    unsigned int testing_items_count    = dataset->get_testing_count();
    unsigned int batch_size             = json_config.result["batch_size"].asInt();
    unsigned int batches_per_epoch      = (dataset->get_training_count()/batch_size);

    if (batches_per_epoch == 0)
        batches_per_epoch = 1;

    std::string best_network_saving_folder = experiment_dir + "trained/";

    Shape input_shape  = dataset->get_input_shape();
    Shape output_shape = dataset->get_output_shape();

    float batch_mem = sizeof(float)*batch_size*(input_shape.size() + output_shape.size())/1000000.0;

    experiment_log << "regression experiment start\n\n";


    experiment_log << "training_items_count " << training_items_count << "\n";
    experiment_log << "testing_items_count  " << testing_items_count << "\n";
    experiment_log << "epoch_count          " << epoch_count << "\n";
    experiment_log << "batch_size           " << batch_size << "\n";
    experiment_log << "batches_per_epoch    " << batches_per_epoch << "\n";
    experiment_log << "input_shape  [" << input_shape.w() << " " << input_shape.h() << " " << input_shape.d() << "]\n";
    experiment_log << "output_shape [" << output_shape.w() << " " << output_shape.h() << " " << output_shape.d() << "]\n";
    experiment_log << "network_config_file  " << network_config_file << "\n";


    experiment_log << s_delimiter;

    experiment_log << "initialising batch with size " << batch_size << "  required memory " << batch_mem << "[MB]" << "\n";

    Batch batch(input_shape, output_shape, batch_size);

    experiment_log << "batch init done\n\n\n";


    experiment_log << "initialising network from file " << network_config_file << "\n";

    CNN cnn(network_config_file, input_shape, output_shape);

    experiment_log << "network init done\n";
    experiment_log << "network init result : \n";
    std::string nn_init_result = cnn.asString();
    experiment_log << nn_init_result << "\n";

    experiment_log << s_delimiter;

    experiment_log << "\n\nstarting training\n\n";



    float rms_best_result = dataset->get_output_shape().size()*10.0;

    for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
        for (unsigned int batch_id = 0; batch_id < batches_per_epoch; batch_id++)
        {
            float training_done = 100.0*(epoch*batches_per_epoch + batch_id)/(epoch_count*batches_per_epoch);

            experiment_log << "creating batch " << batch_id << "\n";
            batch.create(*this->dataset);

            experiment_log << "training epoch " << epoch << " batch id" << batch_id << " done = " << training_done << "[%]" <<"\n\n";

            cnn.train(batch.get_output_all(), batch.get_input_all(), 1, false);

            experiment_log << "testing on testing\n";

            RegressionCompare compare_testing(dataset->get_output_shape().size());

            std::vector<float> nn_output(dataset->get_output_shape().size());

            for (unsigned int item_idx = 0; item_idx < dataset->get_testing_count(); item_idx++)
            {
                cnn.forward(nn_output, dataset->get_testing_input(item_idx));
                compare_testing.add(dataset->get_testing_output(item_idx), nn_output);

                if (compare_testing.is_nan_error())
                {
                    experiment_log << "NaN error\n";
                    break;
                }
            }
            compare_testing.compute();


            experiment_log << "testing on training\n";

            RegressionCompare compare_training(dataset->get_output_shape().size());

            for (unsigned int item_idx = 0; item_idx < dataset->get_testing_count(); item_idx++)
            {
                dataset->set_random_training_idx();

                cnn.forward(nn_output, dataset->get_training_input());
                compare_training.add(dataset->get_training_output(), nn_output);

                if (compare_training.is_nan_error())
                {
                    experiment_log << "NaN error\n";
                    break;
                }
            }
            compare_training.compute();


            experiment_log << "testing error_average_euclidean " << compare_testing.get_error_average_euclidean() << "\n";
            experiment_log << "testing error_std " << compare_testing.get_error_std() << "\n";

            progress_log << epoch << " " << batch_id << " " << training_done << " " <<  compare_testing.get_error_average_euclidean() << " " << compare_testing.get_error_std() << compare_training.get_error_average_euclidean() << " " << compare_training.get_error_std() << "\n";


            if (compare_testing.get_error_average_euclidean() < rms_best_result)
            {
                rms_best_result = compare_testing.get_error_average_euclidean();

                experiment_log << "saving new best net with result = " << rms_best_result << "\n";
                cnn.save(best_network_saving_folder);

                process_best(cnn);
            }

            experiment_log << s_delimiter;
        }

    experiment_log << "training done\n";
}



std::string RegressionExperiment::delimiter()
{
    std::string result;

    for (unsigned int i = 0; i < 100; i++)
        result+= "=";
    result+= "\n\n\n\n\n";

    return result;
}


void RegressionExperiment::process_best(CNN &cnn)
{
    std::vector<float> nn_output(dataset->get_output_shape().size());

    for (unsigned int i = 0; i < 32; i++)
    {
        unsigned item_idx = (i*100)%dataset->get_testing_count();

        auto input = dataset->get_testing_input(item_idx);

        cnn.forward(nn_output, input);

        VectorToImage v_input(input, dataset->get_input_shape());
        VectorToImage v_target(dataset->get_testing_output(item_idx), dataset->get_output_shape());
        VectorToImage v_computed(nn_output, dataset->get_output_shape());

        v_input.save(experiment_dir + "output/" + std::to_string(item_idx) + "_0_input.png");
        v_target.save(experiment_dir + "output/" + std::to_string(item_idx) + "_1_target.png");
        v_computed.save(experiment_dir + "output/" + std::to_string(item_idx) + "_2_computed.png");
    }
}
