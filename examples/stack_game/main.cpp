#include <stack/stack_game.h>
#include <getch.h>
#include <timer.h>
#include <math.h>

#include <dataset_interface.h>
#include <batch.h>
#include <cnn.h>

#include <dataset_stack.h>
#include <classification_compare.h>


void network_train(DatasetInterface *dataset)
{
  unsigned int epoch_count = 10;
  //create and fill batch
  Batch batch(dataset->get_input_size(), dataset->get_output_size(), dataset->get_training_size(), 0.0);

  printf("creating batch with size %u\n", batch.size());

  for (unsigned int i = 0; i < batch.size(); i++)
  {
    sDatasetItem item = dataset->get_training(i);
    batch.add(item.output, item.input);
  }

  sGeometry input_geometry;
  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  sGeometry output_geometry;
  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset->get_output_size();

  INN *nn = new CNN("cnn_config.json", input_geometry, output_geometry);

  printf("learning\n");
  unsigned int training_iterations = batch.size()*epoch_count;

// training_iterations = 100;
  timer.start();
  for (unsigned int i = 0; i < training_iterations; i++)
  {
    batch.set_random();
    nn->learn(batch.get_output(), batch.get_input());

    if ((i%1000) == 0)
      printf("training done %6.3f %%\n", i*100.0/training_iterations);
  }
  timer.stop();

  nn->save("trained/");

  printf("training time %f\n", timer.get_duration()/training_iterations);
}


void network_test(DatasetInterface *dataset)
{
  printf("testing\n");


  sGeometry input_geometry;
  input_geometry.w = dataset->get_width();
  input_geometry.h = dataset->get_height();
  input_geometry.d = dataset->get_channels();

  sGeometry output_geometry;
  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = dataset->get_output_size();

  INN *nn = new CNN("trained/cnn_config.json", input_geometry, output_geometry);


  std::vector<float> nn_output;
  nn_output.resize(dataset->get_output_size());

  ClassificationCompare compare(dataset->get_output_size());

  timer.start();

  for (unsigned int i = 0; i < dataset->get_testing_size(); i++)
  {
    sDatasetItem ground_truth = dataset->testing[i];

    nn->forward(nn_output, ground_truth.input);

    compare.compare(ground_truth.output, nn_output);

    if ((i%100) == 0)
    {
      for (unsigned int j = 0; j < ground_truth.output.size(); j++)
        printf("%6.3f ", ground_truth.output[j]);

      printf(" : ");
      for (unsigned int j = 0; j < nn_output.size(); j++)
        printf("%6.3f ", nn_output[j]);

      printf("\n");
    }
  }
  timer.stop();

  compare.process(true);

  std::string result = compare.get_text_result();

  printf("%s\n", result.c_str());
  printf("forward time %f [ms]\n", timer.get_duration()/dataset->get_testing_size());
}


void stack_manual(unsigned int size)
{
  StackGame stack(size, size);
  Getchar getch;

  int c;

  while ((c = getch) != 27)
  {
    unsigned int action = 0;
    if (c == ' ')
      action = 1;


    stack.do_action(action);
    timer.sleep_ms(100);
  }
}


void stack_nn_bot(unsigned int size)
{
  StackGame stack(size, size);

  sGeometry input_geometry;
  input_geometry.w = stack.get_geometry().w;
  input_geometry.h = stack.get_geometry().h;
  input_geometry.d = stack.get_geometry().d;

  sGeometry output_geometry;
  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = 2;

  INN *nn = new CNN("trained/cnn_config.json", input_geometry, output_geometry);


  std::vector<float> nn_output;
  nn_output.resize(2);


  unsigned int state = 0;
  unsigned int cnt = 0;
  unsigned int try_count = size;
  float confidence_max = 0.0;

  while (1)
  {
    nn->forward(nn_output, stack.get_state());

    float confidence = nn_output[1];

    unsigned int action = 0;

    switch (state)
    {
      case 0:
              if (confidence > confidence_max)
              {
                confidence_max = confidence;
              }

              if (cnt > try_count)
              {
                state = 1;
              //  printf("CM : %f\n", confidence_max);
              }
              else
                cnt++;


              break;

        case 1:
              if (fabs(confidence_max - confidence) < 0.3)
              {
                  // printf(">>> : %f\n\n", confidence);

                  action = 1;
                  confidence = 0.0;
                  confidence_max = 0.0;
                  cnt = 0;
                  state = 0;
              }

              break;
    }

    stack.do_action(action);

    timer.sleep_ms(5);
  }
}


int main(int argc, char *argv[])
{
  srand(time(NULL));

  unsigned int size = 32;

  if (argc == 2)
  {
    if (argv[1][0] == 'r')
      stack_nn_bot(size);

    if (argv[1][0] == 't')
    {
      unsigned int training_count = 100000;
      unsigned int testing_count  = 1000;
      unsigned int unlabeled_count  = 0;

      DatasetStack dataset(size, size, training_count, testing_count, unlabeled_count, 0.0, 0.0, 0.5);


      network_train(&dataset);
      network_test(&dataset);

      printf ("program done\n");
    }
  }
  else
   stack_manual(size);

  return 0;
}
