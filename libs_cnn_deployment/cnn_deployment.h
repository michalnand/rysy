#ifndef _CNN_Deployment_H_
#define _CNN_Deployment_H_

#include <layer_interface.h>

class CNNDeployment
{
    public:
        CNNDeployment();
        CNNDeployment(CNNDeployment& other);
        CNNDeployment(const CNNDeployment& other);

        CNNDeployment(std::string json_file_name, sGeometry input_geometry = {0, 0, 0});

        virtual ~CNNDeployment();

        CNNDeployment& operator= (CNNDeployment& other);
        CNNDeployment& operator= (const CNNDeployment& other);

    protected:
        void copy(CNNDeployment& other);
        void copy(const CNNDeployment& other);

    public:
        void init(std::string json_file_name, sGeometry input_geometry = {0, 0, 0});

        sGeometry get_input_geometry();
        sGeometry get_output_geometry();

        unsigned int get_input_size();
        unsigned int get_output_size();

        void forward(std::vector<float> &output, std::vector<float> &input);

    private:
        sGeometry input_geometry, output_geometry;
        float *buffer_a, *buffer_b;

        std::vector<LayerInterface*> layers;

    private:
        LayerInterface* create_layer(Json::Value json, sGeometry input_geometry);

};

#endif