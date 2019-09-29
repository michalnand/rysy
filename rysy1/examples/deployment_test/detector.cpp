#include <detector.h>
#include <timer.h>
#include <iostream>

Detector::Detector(std::string network_config_file_name, unsigned int image_width, unsigned int image_height, float confidence)
{
    JsonConfig json(network_config_file_name);

    unsigned int last_layer = json.result["layers"].size() - 1;

    this->image_width = image_width;
    this->image_height = image_height;
    this->confidence = confidence;

    sGeometry input_geometry;
    input_geometry.w = image_width;
    input_geometry.h = image_height;
    input_geometry.d = 3;

    sGeometry output_geometry;

    unsigned int output_kernel_width    = json.result["layers"][last_layer]["input_geometry"][0].asInt();
    unsigned int output_kernel_height   = json.result["layers"][last_layer]["input_geometry"][1].asInt();
    unsigned int output_kernel_depth    = json.result["layers"][last_layer]["output_geometry"][2].asInt();

    width_ratio    = json.result["input_geometry"][0].asInt()/output_kernel_width;
    height_ratio   = json.result["input_geometry"][1].asInt()/output_kernel_height;

    output_geometry.w  = output_kernel_width;
    output_geometry.h  = output_kernel_height;
    output_geometry.d  = output_kernel_depth;

    output_width    = input_geometry.w/width_ratio;
    output_height   = input_geometry.h/height_ratio;
    output_depth    = output_kernel_depth;

    cnn = new CNNDeployment(network_config_file_name, input_geometry);

    result_init();

    color_palette = generate_color_palette(output_depth - 1);
}

Detector::~Detector()
{
    delete cnn;
}

float Detector::cnn_output_get(unsigned int x, unsigned y, unsigned ch)
{
    unsigned int idx = (ch*output_height + y)*output_width + x;
    return cnn_output[idx];
}

void Detector::process(std::vector<float> &image_v)
{
    Timer timer;

    timer.start();

    cnn->forward(cnn_output, image_v);

    unsigned int ptr;
    ptr = 0;

    int y_shift = 2;
    int x_shift = 2;


    for (unsigned int j = 0; j < output_height - y_shift; j++)
    for (unsigned int i = 0; i < output_width - x_shift; i++)
    {
        unsigned int max_k = 0;
        for (unsigned int k = 0; k < output_depth; k++)
        {
            float conf_best = cnn_output_get(i, j, max_k);
            float conf = cnn_output_get(i, j, k);

            if (k != 0)
            if (conf > conf_best)
            if (conf > confidence)
            {
                conf_best = conf;
                max_k = k;
            }
        }

        result.class_result[j + y_shift][i + x_shift] = max_k;
    }




    timer.stop();
    result.computing_time = timer.get_duration();


    result.json.clear();

    unsigned int classes_count = output_depth;

    result.json["width"] = output_width;
    result.json["output_height"] = output_height;
    result.json["classes_count"] = classes_count;
    result.json["computing_time"] = result.computing_time;

    unsigned int result_idx = 0;
    for (unsigned int j = 0; j < result.class_result.size(); j++)
    for (unsigned int i = 0; i < result.class_result[j].size(); i++)
    {
        unsigned int class_id = result.class_result[j][i];
        if (class_id != 0)
        {
            result.json["result"][result_idx][0] = class_id;
            result.json["result"][result_idx][1] = i;
            result.json["result"][result_idx][2] = j;
            result_idx++;
        }
    }

    Json::FastWriter fastWriter;
    result.json_string = fastWriter.write(result.json);


}

void Detector::process(cv::Mat &image)
{
    unsigned int layer_size = image_width*image_height;
    unsigned int input_idx = 0;

    for (unsigned int y = 0; y < image_height; y++)
        for (unsigned int x = 0; x < image_width; x++)
        {
            auto pixel = image.at<cv::Vec3b>(y,x);
            float r = pixel[2]/256.0;
            float g = pixel[1]/256.0;
            float b = pixel[0]/256.0;


            cnn_input[input_idx + 0*layer_size] = r;
            cnn_input[input_idx + 1*layer_size] = g;
            cnn_input[input_idx + 2*layer_size] = b;

            input_idx++;
        }

    process(cnn_input);
}


sDetectorResult& Detector::get_result()
{
    return result;
}





void Detector::inpaint_class_result(std::vector<float> &image_v, float alpha)
{
    unsigned int padding = 16;

    for (unsigned int k = 0; k < 3; k++)
    for (unsigned int j = padding; j < image_height - padding; j++)
    for (unsigned int i = padding; i < image_width - padding; i++)
    {
        unsigned int idx = (k*image_height + j)*image_width + i;
        unsigned int res_j = (j)/height_ratio;
        unsigned int res_i = (i)/width_ratio;

        unsigned int class_id = result.class_result[res_j][res_i];

        if (class_id != 0)
        {
            float v = alpha*image_v[idx] + (1.0 - alpha)*get_class_color(class_id-1)[k];
            image_v[idx] = v;
        }
    }
}


void Detector::inpaint_class_result(cv::Mat &image, float alpha)
{
    inpaint_class_result(cnn_input, alpha);

    unsigned int layer_size = image_width*image_height;
    unsigned int input_idx = 0;

    for (unsigned int y = 0; y < image_height; y++)
        for (unsigned int x = 0; x < image_width; x++)
        {
            float r = cnn_input[input_idx + 0*layer_size];
            float g = cnn_input[input_idx + 1*layer_size];
            float b = cnn_input[input_idx + 2*layer_size];

            image.at<cv::Vec3b>(y,x)[2] = r*255;
            image.at<cv::Vec3b>(y,x)[1] = g*255;
            image.at<cv::Vec3b>(y,x)[0] = b*255;

            input_idx++;
        }
}

void Detector::result_init()
{
    result.output_width     = output_width;
    result.output_height    = output_height;
    result.classes_count    = output_depth;

    unsigned int input_size = image_width*image_height*3;
    cnn_input.resize(input_size);
    for (unsigned int i = 0; i < cnn_input.size(); i++)
        cnn_input[i] = 0.0;


    unsigned int output_size = output_width*output_height*output_depth;
    cnn_output.resize(output_size);
    for (unsigned int i = 0; i < cnn_output.size(); i++)
        cnn_output[i] = 0.0;


    result.class_result.resize(output_height);
    for (unsigned int j = 0; j < output_height; j++)
    {
        result.class_result[j].resize(output_width);
        for (unsigned int i = 0; i < output_width; i++)
            result.class_result[j][i] = 0;
    }

    result.json.clear();
}

std::vector<std::vector<float>> Detector::generate_color_palette(unsigned int count)
{
    std::vector<std::vector<float>> result(count);

    for (unsigned int i = 0; i < count; i++)
    {
        float c = 2.0*3.141592654;
        float phase = i*c/count;

        result[i].push_back((sin(phase + c*0.0*1.0/3.0) + 1.0)/2.0);
        result[i].push_back((sin(phase + c*1.0*1.0/3.0) + 1.0)/2.0);
        result[i].push_back((sin(phase + c*2.0*1.0/3.0) + 1.0)/2.0);
    }

    return result;
}

std::vector<float>& Detector::get_class_color(unsigned int class_id)
{
    return color_palette[class_id];
}
