#pragma version(1)
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

float2 __attribute__((kernel)) root(uint32_t x)
{
    float2 sum = 0;
    sum.x = sum.y = 0;

    int kernel_num = x % (n_k / 2);


    int h_num = (x * 2) / (w_o * n_k);
    int w_num = (x % (w_o * n_k / 2)) / (n_k / 2);

    int g = (kernel_num * 2) / (n_k / group);
    int channel_offset = g * c_k / 4;

    int c_k_new = c_k / 4;

    for (int h = 0 ; h < h_k ; h++){
        for (int w = 0 ; w < w_k ; w++){
            for (int i = 0 ; i < c_k_new / 2 ; i++)
            {
                int cur_x = h_num * stride_x + h;           //should take care of the strides(Be careful)
                int cur_y = w_num * stride_y + w;           //should take care of the strides(Be careful)

                if (cur_x < pad_x || cur_x >= (pad_x + h_i))
                    continue;
                else if (cur_y < pad_y || cur_y >= (pad_y + w_i))
                    continue;
                else
                {
                    int frame_index = (cur_x - pad_x) * w_i * c_i / 4 + (cur_y - pad_y) * c_i / 4 + (2 * i + channel_offset);
                    float4 frame_value1 = rsGetElementAt_float4(In_Blob,frame_index);
					float4 frame_value2 = rsGetElementAt_float4(In_Blob,frame_index + 1);

                    float4 kernel_value1, kernel_value2;
                    int kernel_size = h_k * w_k * c_k_new;
                    int kernel_index = kernel_num * 2 * kernel_size +  h * w_k * c_k_new + w *  c_k_new + 2 * i;

                    kernel_value1 = rsGetElementAt_float4(Kernel_Blob,kernel_index);
                    kernel_value2 = rsGetElementAt_float4(Kernel_Blob,kernel_index + 1);
                    sum.x += dot(frame_value1 ,kernel_value1) + dot(frame_value2 ,kernel_value2);

                    kernel_index += kernel_size;
                    kernel_value1 = rsGetElementAt_float4(Kernel_Blob,kernel_index);
                    kernel_value2 = rsGetElementAt_float4(Kernel_Blob,kernel_index + 1);
                    sum.y += dot(frame_value1 ,kernel_value1) + dot(frame_value2 ,kernel_value2);
                }
            }
        }
    }
	
    return sum + rsGetElementAt_float2(Bias_Blob,kernel_num);
}