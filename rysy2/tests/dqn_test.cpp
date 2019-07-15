#include <iostream>
#include <dqn.h>

int main()
{
    Shape state_shape(80, 80, 24);
    unsigned int actions_count = 3;

    DQN dqn(state_shape, actions_count);

    dqn.add_layer("convolution", Shape(3, 3, 32));
    dqn.add_layer("elu");
    dqn.add_layer("max pooling", Shape(2, 2, 1));

    dqn.add_layer("convolution", Shape(3, 3, 64));
    dqn.add_layer("elu");
    dqn.add_layer("max pooling", Shape(2, 2, 1));

    dqn.add_layer("convolution", Shape(3, 3, 64));
    dqn.add_layer("elu");
    dqn.add_layer("max pooling", Shape(2, 2, 1));

    dqn.add_layer("convolution", Shape(3, 3, 64));
    dqn.add_layer("elu");
    dqn.add_layer("max pooling", Shape(2, 2, 1));

    dqn.add_layer("fc", Shape(1, 1, 256));
    dqn.add_layer("elu");

    dqn.add_layer("output");

    std::vector<float> state(state_shape.size());

    for (unsigned int i = 0; i < state.size(); i++)
        state[i] = (rand()%1000000)/1000000.0;

    auto q_values = dqn.forward(state);


    for (unsigned int i = 0; i < q_values.size(); i++)
        std::cout << q_values[i] << " ";
    std::cout << "\n";



    std::cout << "program done\n";
    return 0;
}
