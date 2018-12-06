
#define  kernels_count 32
#define  kh 5
#define  kw 5

#define width           256
#define height          256
#define channels_count  3


void back_kernel()
{
  float weights[kernels_count][channels_count][kh][kw];
  float error[kernels_count][height][width];
  float error_back[channels_count][height][width];

  for (unsigned int ch = 0; ch < channels_count; ch++)
  for (unsigned int y = 0; y < height; y++)
  for (unsigned int x = 0; x < width; x++)
  {
    float sum = 0.0;
    for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
      for (unsigned int ky = 0; ky < kh; ky++)
        for (unsigned int kx = 0; kx < kw; kx++)
        {
          sum+= error[kernel][ky + y][kw + x]*weights[kernel][ch][ky][kx];
        }

    error_back[ch][y][x] = sum;
  }
}
