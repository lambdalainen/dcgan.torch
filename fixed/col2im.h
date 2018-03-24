#ifndef __COL2IM_H__
#define __COL2IM_H__

void col2im(const float* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            float* data_im);

#endif
