#include <export_stl10.h>
#include <fstream>
#include <image_save.h>



ExportSTL10::ExportSTL10(std::string input_file_name, std::string output_dir)
{
    this->input_file_name = input_file_name;
    this->output_dir = output_dir;
}

ExportSTL10::~ExportSTL10()
{

}


void ExportSTL10::process(unsigned int max_count)
{
    unsigned int width = 96;
    unsigned int height = 96;
    unsigned int channels = 3;

    unsigned int size = width*height*channels;

    std::vector<unsigned char> raw_data(size);
    std::vector<float> output(size);

    std::ifstream file(input_file_name, std::ios::in | std::ios::binary);

    ImageSave image(width, height, false);

    for (unsigned int i = 0; i < max_count; i++)
    {
        std::string output_file_name = output_dir + std::to_string(i) + ".png";

        file.read((char*)(&raw_data[0]), raw_data.size());


        for (unsigned int j = 0; j < output.size(); j++)
            output[j] = raw_data[j];

        image.save(output_file_name, output);
    }

    file.close();
}
