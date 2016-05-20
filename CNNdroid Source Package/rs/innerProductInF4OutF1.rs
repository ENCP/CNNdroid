#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation In_Blob;
rs_allocation Kernel_Blob;
rs_allocation Bias_Blob;
int c_i;
int w_w;
int c_o;

float __attribute__((kernel)) root(uint32_t x)
{
    float sum = 0;
    int pic_num = x / c_o;
	int kernel_num = x % c_o;

	int kernel_offset = kernel_num * w_w / 4;
	int frame_offset = pic_num * w_w / 4;

    for (int i = 0 ; i < c_i / 4 ; i++){
        int frame_index = frame_offset + i;
        int kernel_index = kernel_offset + i;
        float4 frame_value = rsGetElementAt_float4(In_Blob,frame_index);
        float4 kernel_value = rsGetElementAt_float4(Kernel_Blob,kernel_index);
        sum += dot(frame_value, kernel_value);
    }
    return sum + rsGetElementAt_float(Bias_Blob, kernel_num);
}