#pragma version(1)
#pragma rs java_package_name(layers)

rs_allocation frames;
rs_allocation kernel;
int c_k, h_k, w_k;
int h_i, w_i;
int w_o;
int h_s, w_s;
int pad_x, pad_y;
float bias;


float __attribute__((kernel)) root(uint32_t x)
{
    float sum = 0;
    int a = (x / w_o) * h_s;
    int b = (x % w_o) * w_s;

    for (int i = 0; i < c_k; ++i)
        for (int j = 0; j < h_k; ++j)
            for (int k = 0; k < w_k; ++k)
            {
                float frame_value;
                int cur_x = a + j;
                int cur_y = b + k;
                if (cur_x < pad_x || cur_x >= (pad_x + h_i))
                    frame_value = 0;
                else if (cur_y < pad_y || cur_y >= (pad_y + w_i))
                    frame_value = 0;
                else
                    frame_value = rsGetElementAt_float(frames, i * h_i * w_i + (cur_x - pad_x) * w_i + (cur_y - pad_y));

                sum += frame_value * rsGetElementAt_float(kernel, i * h_k * w_k + j * w_k + k);
            }

    return sum + bias;
}