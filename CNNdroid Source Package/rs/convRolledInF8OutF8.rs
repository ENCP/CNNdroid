#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation In_Blob;
rs_allocation Kernel_Blob;
rs_allocation Bias_Blob;
rs_allocation Out_Alloc;
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

void root(float4* out, uint32_t x)
{
    float4 sum1, sum2;
    sum1.x = sum1.y = sum1.z = sum1.w = 0;
    sum2.x = sum2.y = sum2.z = sum2.w = 0;

    int kernel_num = x % (n_k / 8);

    int g = (kernel_num * 8) / (n_k / group);
    int channel_offset = g * c_k / 4;

    int h_num = (x * 8) / (w_o * n_k);
    int w_num = (x % (w_o * n_k / 8)) / (n_k / 8);

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

                    float4 kernel_value11, kernel_value12, kernel_value21, kernel_value22;
                    int kernel_size = h_k * w_k * c_k_new;
                    int kernel_offset = 2 * kernel_size;
                    int kernel_index = kernel_num * 8 * kernel_size +  h * w_k * c_k_new + w *  c_k_new + 2 * i;

                    kernel_value11 = rsGetElementAt_float4(Kernel_Blob, kernel_index);
                    kernel_value12 = rsGetElementAt_float4(Kernel_Blob, kernel_index + 1);
                    kernel_value21 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size);
                    kernel_value22 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size + 1);
                    sum1.x += dot(frame_value1, kernel_value11) + dot(frame_value2, kernel_value12);
                    sum2.x += dot(frame_value1, kernel_value21) + dot(frame_value2, kernel_value22);


                    kernel_index += kernel_offset;
                    kernel_value11 = rsGetElementAt_float4(Kernel_Blob, kernel_index);
                    kernel_value12 = rsGetElementAt_float4(Kernel_Blob, kernel_index + 1);
                    kernel_value21 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size);
                    kernel_value22 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size + 1);
                    sum1.y += dot(frame_value1, kernel_value11) + dot(frame_value2, kernel_value12);
                    sum2.y += dot(frame_value1, kernel_value21) + dot(frame_value2, kernel_value22);


                    kernel_index += kernel_offset;
                    kernel_value11 = rsGetElementAt_float4(Kernel_Blob, kernel_index);
                    kernel_value12 = rsGetElementAt_float4(Kernel_Blob, kernel_index + 1);
                    kernel_value21 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size);
                    kernel_value22 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size + 1);
                    sum1.z += dot(frame_value1, kernel_value11) + dot(frame_value2, kernel_value12);
                    sum2.z += dot(frame_value1, kernel_value21) + dot(frame_value2, kernel_value22);


                    kernel_index += kernel_offset;
                    kernel_value11 = rsGetElementAt_float4(Kernel_Blob, kernel_index);
                    kernel_value12 = rsGetElementAt_float4(Kernel_Blob, kernel_index + 1);
                    kernel_value21 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size);
                    kernel_value22 = rsGetElementAt_float4(Kernel_Blob, kernel_index + kernel_size + 1);
                    sum1.w += dot(frame_value1, kernel_value11) + dot(frame_value2, kernel_value12);
                    sum2.w += dot(frame_value1, kernel_value21) + dot(frame_value2, kernel_value22);

                }
            }
        }
    }

    float4 bias1 = rsGetElementAt_float4(Bias_Blob,kernel_num * 2);
    float4 bias2 = rsGetElementAt_float4(Bias_Blob,kernel_num * 2 + 1);

	sum1.x += bias1.x;
	sum2.x += bias1.y;

	sum1.y += bias1.z;
	sum2.y += bias1.w;

	sum1.z += bias2.x;
	sum2.z += bias2.y;

	sum1.w += bias2.z;
	sum2.w += bias2.w;

	rsSetElementAt_float4(Out_Alloc, sum2, x);

	(*out) = sum1;
}