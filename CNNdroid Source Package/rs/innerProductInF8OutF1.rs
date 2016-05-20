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

    for (int i = 0 ; i < c_i / 8 ; i++){
        int frame_index1 = frame_offset + 2 * i;
		int frame_index2 = frame_offset + 2 * i + 1;
        float4 frame_value1 = rsGetElementAt_float4(In_Blob,frame_index1);
		float4 frame_value2 = rsGetElementAt_float4(In_Blob,frame_index2);
		int kernel_index1 = kernel_offset + 2 * i;
		int kernel_index2 = kernel_offset + 2 * i + 1;
        float4 kernel_value1 = rsGetElementAt_float4(Kernel_Blob,kernel_index1);
		float4 kernel_value2 = rsGetElementAt_float4(Kernel_Blob,kernel_index2);
        sum += dot(frame_value1, kernel_value1) + dot(frame_value2, kernel_value2);
    }
    return sum + rsGetElementAt_float(Bias_Blob, kernel_num);
}