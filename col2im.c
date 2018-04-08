#include <string.h>
#include "col2im.h"

void col2im(const float* data_col, const int nOutputPlane,
      const int outputHeight, const int outputWidth,
      const int inputHeight, const int inputWidth,
      const int kH, const int kW,
      const int padH, const int padW,
      const int strideH, const int strideW,
      const int dilationH, const int dilationW,
      float* data_im)
{
  memset(data_im, 0, sizeof(float) * outputHeight * outputWidth * nOutputPlane);
  const int n = nOutputPlane * kH * kW;
  for (int j = 0; j < n; ++j) {
    int w_offset = j % kW;
    int h_offset = (j / kW) % kH;
    int c_im = j / kH / kW;
    for (int h_col = 0; h_col < inputHeight; ++h_col) {
      for (int w_col = 0; w_col < inputWidth; ++w_col) {
        int h_im = h_col * strideH - padH + h_offset * dilationH;
        int w_im = w_col * strideW - padW + w_offset * dilationW;
        if (h_im >= 0 && h_im < outputHeight && w_im >= 0 && w_im < outputWidth)
          data_im[(c_im * outputHeight + h_im) * outputWidth + w_im] +=
            data_col[(j * inputHeight + h_col) * inputWidth + w_col];
      }
    }
  }
}
