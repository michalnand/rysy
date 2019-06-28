#include <classification_experiment.h>
#include <iostream>

#include <batch.h>
#include <cnn.h>



ClassificationExperiment::ClassificationExperiment(DatasetInterface &dataset, std::string experiment_dir, std::string network_config_file)
{
    this->dataset = &dataset;
    this->experiment_dir      = experiment_dir;
    this->network_config_file = experiment_dir + network_config_file;
}

ClassificationExperiment::~ClassificationExperiment()
{

}

void ClassificationExperiment::run()
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

    experiment_log << "classification experiment start\n\n";


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

    float accuracy_result_best = 0.0;

    for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
        for (unsigned int batch_id = 0; batch_id < batches_per_epoch; batch_id++)
        {
            float training_done = 100.0*(epoch*batches_per_epoch + batch_id)/(epoch_count*batches_per_epoch);

            experiment_log << "creating batch " << batch_id << "\n";
            batch.create(*this->dataset);

            experiment_log << "training epoch " << epoch << " batch id" << batch_id << " done = " << training_done << "[%]" <<"\n\n";

            cnn.train(batch.get_output_all(), batch.get_input_all(), 1, false);

            experiment_log << "testing on testing\n";

            compare_testing.init(dataset->get_classes_count());
            compare_testing_top5.init(dataset->get_classes_count(), 5);

            std::vector<float> nn_output(dataset->get_classes_count());

            for (unsigned int item_idx = 0; item_idx < dataset->get_testing_count(); item_idx++)
            {
                cnn.forward(nn_output, dataset->get_testing_input(item_idx));

                int compare_testing_result = compare_testing.add(dataset->get_testing_output(item_idx), nn_output);
                int compare_testing_top_5_result = compare_testing_top5.add(dataset->get_testing_output(item_idx), nn_output);

                if (compare_testing_result != 0)
                {
                    experiment_log << "compare testing error " << compare_testing_result << "\n";
                    break;
                }

                if (compare_testing_top_5_result != 0)
                {
                    experiment_log << "compare testing TOP5 error " << compare_testing_top_5_result << "\n";
                    break;
                }
            }
            compare_testing.compute();
            compare_testing_top5.compute();


            experiment_log << "testing on training\n";

            compare_training.init(dataset->get_classes_count());
            compare_training_top5.init(dataset->get_classes_count(), 5);

            for (unsigned int item_idx = 0; item_idx < dataset->get_testing_count(); item_idx++)
            {
                dataset->set_random_training_idx();

                cnn.forward(nn_output, dataset->get_training_input());

                int compare_training_result = compare_training.add(dataset->get_training_output(), nn_output);
                int compare_training_top_5_result = compare_training_top5.add(dataset->get_training_output(), nn_output);

                if (compare_training_result != 0)
                {
                    experiment_log << "compare training error " << compare_training_result << "\n";
                    break;
                }

                if (compare_training_top_5_result != 0)
                {
                    experiment_log << "compare training TOP5 error " << compare_training_top_5_result << "\n";
                    break;
                }
            }
            compare_training.compute();
            compare_training_top5.compute();


            experiment_log << "testing accuracy " << compare_testing.get_accuracy() << "[%]\n";
            experiment_log << "testing accuracy TOP 5 " << compare_testing_top5.get_accuracy() << "[%]\n";

            experiment_log << "training accuracy   " << compare_training.get_accuracy() << "[%]\n";
            experiment_log << "training accuracy TOP 5 " << compare_training_top5.get_accuracy() << "[%]\n";

            progress_log << epoch << " " << batch_id << " " << training_done << " " << compare_testing.get_accuracy() << " " << compare_training.get_accuracy() << " " << compare_testing_top5.get_accuracy() << " " << compare_training_top5.get_accuracy() << "\n";


            if (compare_testing.get_accuracy() > accuracy_result_best)
            {
                accuracy_result_best = compare_testing.get_accuracy();

                experiment_log << "saving new best net with result = " << accuracy_result_best << "%\n";
                cnn.save(best_network_saving_folder);

                Log confussion_matrix_log(experiment_dir + "confussion_matrix.log");

                confussion_matrix_log << "testing accuracy " << compare_testing.get_accuracy() << "[%]\n";
                confussion_matrix_log << "testing accuracy TOP 5 " << compare_testing_top5.get_accuracy() << "[%]\n";
                confussion_matrix_log << "training accuracy   " << compare_training.get_accuracy() << "[%]\n";
                confussion_matrix_log << "training accuracy TOP 5 " << compare_training_top5.get_accuracy() << "[%]\n";

                confussion_matrix_log << "\n\n\n";

                std::string test_cm = compare_testing.asString();
                confussion_matrix_log << "TESTING confussion matrix : \n\n";
                confussion_matrix_log << test_cm;
                confussion_matrix_log << s_delimiter;

                std::string train_cm = compare_training.asString();
                confussion_matrix_log << "TRAINING confussion matrix : \n\n";
                confussion_matrix_log << train_cm;
                confussion_matrix_log << s_delimiter;


                std::string test_cm_top_5 = compare_testing_top5.asString();
                confussion_matrix_log << "TESTING TOP 5 confussion matrix : \n\n";
                confussion_matrix_log << test_cm_top_5;
                confussion_matrix_log << s_delimiter;

                std::string train_cm_top_5 = compare_training_top5.asString();
                confussion_matrix_log << "TRAINING TOP 5 confussion matrix : \n\n";
                confussion_matrix_log << train_cm_top_5;
                confussion_matrix_log << s_delimiter;


                JsonConfig testing_best;
                testing_best.result = compare_testing.asJson();
                testing_best.save(experiment_dir + "testing_best.json");

                JsonConfig training_best;
                training_best.result = compare_training.asJson();
                training_best.save(experiment_dir + "training_best.json");

                JsonConfig testing_best_top_5;
                testing_best_top_5.result = compare_testing_top5.asJson();
                testing_best_top_5.save(experiment_dir + "testing_best_top_5.json");

                JsonConfig training_best_top5;
                training_best_top5.result = compare_training_top5.asJson();
                training_best_top5.save(experiment_dir + "training_best_top5.json");

                process_best();
            }

            experiment_log << s_delimiter;
        }

    experiment_log << "training done\n";
}

std::string ClassificationExperiment::delimiter()
{
    std::string result;

    for (unsigned int i = 0; i < 100; i++)
        result+= "=";
    result+= "\n\n\n\n\n";

    return result;
}

void ClassificationExperiment::process_best()
{

}
