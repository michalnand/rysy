#ifndef _NETWORK_ACTIVITY_H_
#define _NETWORK_ACTIVITY_H_

#include <cnn.h>
#include <tensor_to_activity_map.h>


class NetworkActivity
{
    public:
        NetworkActivity(CNN &cnn);
        virtual ~NetworkActivity();

        void add();
        void save(std::string output_path);

    private:
        CNN *cnn;

        unsigned int size_min, size_max;
        std::vector<unsigned int> layer_idx;
        std::vector<TensorToActivityMap> activity_maps;
};

#endif
