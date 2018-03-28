#include <stdio.h>
#include <string.h>
#include "../col2im.h"

void col2im(const float* data_col, const int channels,
      const int height, const int width,
      const int output_height, const int output_width,
      const int kernel_h, const int kernel_w,
      const int pad_h, const int pad_w,
      const int stride_h, const int stride_w,
      const int dilation_h, const int dilation_w,
      float* data_im)
{
  memset(data_im, 0, sizeof(float) * height * width * channels);
  const int height_col = output_height;
  const int width_col = output_width;
  const int channels_col = channels * kernel_h * kernel_w;
  printf(">>> height_col %i, width_col %i, channels_col %i\n",
          height_col, width_col, channels_col);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    printf(">>> c_col %i, w_offset %i, h_offset %i, c_im %i\n",
            c_col, w_offset, h_offset, c_im);
    for (int h_col = 0; h_col < height_col; ++h_col) {
      printf(">>>     h_col %i\n", h_col);
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
        int w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        printf(">>>         w_col %i, h_im %i, w_im %i\n", w_col, h_im, w_im);
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
          printf(">>>         data_im[%i] += data_col[%i] (%.2f)\n",
                  (c_im * height + h_im) * width + w_im,
                  (c_col * height_col + h_col) * width_col + w_col,
                  data_col[(c_col * height_col + h_col) * width_col + w_col]);
          data_im[(c_im * height + h_im) * width + w_im] +=
            data_col[(c_col * height_col + h_col) * width_col + w_col];
        }
      }
    }
  }
}
