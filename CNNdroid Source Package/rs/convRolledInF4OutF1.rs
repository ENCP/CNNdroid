#pragma version(1)
#pragma rs_fp_relaxed
#pragma rs java_package_name(layers)

rs_allocation In_Blob;
rs_allocation Kernel_Blob;
rs_allocation Bias_Blob;

int c_i;
int h_i;
int w_i;
int n_k;
int c_k;
int h_k;
int w_k;
int h_o;
int w_o;
int pad_x;
int pad_y;
int stride_x;
int stride_y;
int group;

float __attribute__((kernel)) root(uint32_t x)
{
    float sum = 0;
    int kernel_num = x / (h_o * w_o);

    int g = kernel_num / (n_k / group);
    int channel_offset = g * c_k / 4;

    int h_num = (x % (h_o * w_o)) / w_o;
    int w_num = x % w_o;

    int c_k_new = c_k / 4;

    for (int h = 0 ; h < h_k ; h++){
        for (int w = 0 ; w < w_k ; w++){
            for (int i = 0 ; i < c_k_new ; i++)
            {
                int cur_x = h_num * stride_x + h;           //should take care of the strides(Be careful)
                int cur_y = w_num * stride_y + w;           //should take care of the strides(Be careful)

                if (cur_x < pad_x || cur_x >= (pad_x + h_i))
                    continue;
                else if (cur_y < pad_y || cur_y >= (pad_y + w_i))
                    continue;
                else
                {
                    int frame_index = (cur_x - pad_x) * w_i * c_i / 4 + (cur_y - pad_y) * c_i / 4 + (i + channel_offset);
                    float4 frame_value = rsGetElementAt_float4(In_Blob,frame_index);

                    int kernel_index = kernel_num * h_k * w_k * c_k_new +  h * w_k * c_k_new + w *  c_k_new + i;
                    float4 kernel_value = rsGetElementAt_float4(Kernel_Blob,kernel_index);

                    sum += dot(frame_value,kernel_value);
                }
            }
        }
    }
	
    return sum + rsGetElementAt_float(Bias_Blob,kernel_num);
}