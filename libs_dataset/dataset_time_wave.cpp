#include <dataset_time_wave.h>
#include <math.h>
#include <iostream>

DatasetTimeWave::DatasetTimeWave(unsigned int items_count, unsigned int time_window_size, bool rnn_mode)
                :TimeSeriesDatasetInterface()
{
    if (rnn_mode)
        width   = 1;
    else
        width   = time_window_size;

    height       = 1;
    channels     = 1;

    output_size = 2;
    training.resize(output_size);

    waves_count = 10;

    for (unsigned int i = 0; i < waves_count; i++)
    {
        a.push_back(16.0*(rand()%10000)/10000.0);
        b.push_back(3.141592654*2.0*(rand()%10000)/10000.0);
        c.push_back(16.0*(rand()%10000)/10000.0);
        d.push_back(3.141592654*2.0*(rand()%10000)/10000.0);
    }

    float testing_ratio = 0.1;

    for (unsigned int i = 0; i < items_count; i++)
    {
        auto item = create_item(time_window_size);
        float p = (rand()%10000)/10000.0;

        if (p < testing_ratio)
            add_testing(item);
        else
            add_training(item);
    }

    /*
    for (unsigned int j = 0; j < 10; j++)
    {
        sDatasetItem item = get_random_testing();

        for (unsigned int i = 0; i < item.input.size(); i++)
            std::cout << item.input[i] << " ";

        std::cout << " : ";

        for (unsigned int i = 0; i < item.output.size(); i++)
            std::cout << item.output[i] << " ";

        std::cout << "\n";
    }
    */

    print();
}




DatasetTimeWave::~DatasetTimeWave()
{

}

sDatasetItem DatasetTimeWave::create_item(unsigned int time_window_size)
{
    sDatasetItem item;

    float x     = 1000.0*(rand()%1000000)/1000000.0;
    float dx    = 1.0;
    for (unsigned int t = 0; t < time_window_size; t++)
    {
        float y = func(x);

        item.input.push_back(y);

        x+= dx;
    }

    float target = func(x + 100*dx);

    if (target > 0.0)
    {
        item.output.push_back(1.0);
        item.output.push_back(0.0);
    }
    else
    {
        item.output.push_back(0.0);
        item.output.push_back(1.0);
    }

    return item;
}

float DatasetTimeWave::func(float x)
{
    float result = 0.0;
    for (unsigned int i = 0; i < waves_count; i++)
    {
        result+= sin(a[i]*x + b[i])*sin(c[i]*x + d[i]);
    }

    return result;
}
