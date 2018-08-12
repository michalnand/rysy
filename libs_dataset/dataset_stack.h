#ifndef _STACK_H_
#define _STACK_H_

#include "dataset_interface.h"


class DatasetStack: public DatasetInterface
{
  private:
    float noise;
    float noise_background;
    float alignment_probability;

  public:

    DatasetStack( unsigned int width,
                  unsigned int height,
                  unsigned int training_count,
                  unsigned int testing_count,
                  unsigned int unlabeled_count,
                  float noise = 0.0,
                  float noise_background = 0.0,
                  float alignment_probability = 0.5);

    ~DatasetStack();


  protected:
    void save(std::string file_name_prefix, std::vector<struct sDatasetItem> &items);

    sDatasetItem create_item(std::string *item_image_file_name = nullptr);

    std::vector<std::vector<float>> get_noise_mask(unsigned int rect_w, unsigned int rect_h, float noise);
    std::vector<std::vector<float>> random_rectangle( unsigned int rect_w,
                                                      unsigned int rect_h,
                                                      int x,
                                                      int y,
                                                      std::vector<std::vector<float>> &noise_mask,
                                                      std::vector<std::vector<float>> &background_mask);

    void print_rectangle(std::vector<std::vector<float>> &rect);

    float compute_confidence(std::vector<std::vector<float>> &rect_ref, std::vector<std::vector<float>> &rect_test);

    void save_image(std::string item_image_file_name, std::vector<std::vector<float>> &rectangle_a, std::vector<std::vector<float>> &rectangle_b);

};

#endif
