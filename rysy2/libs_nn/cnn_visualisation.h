#ifndef _CNN_VISUALSIATION_H_
#define _CNN_VISUALSIATION_H_

#include <cnn.h>
#include <tensor_to_image.h>

class CNNVisualisation
{
    public:
        CNNVisualisation(CNN &nn);
        virtual ~CNNVisualisation();

        void process();
        void save(std::string file_name_prefix);

    private:
        bool correct_shape(Shape shape);


    protected:
        CNN *nn;
        std::vector<TensorToImage*> tensor_to_image;
};

#endif
