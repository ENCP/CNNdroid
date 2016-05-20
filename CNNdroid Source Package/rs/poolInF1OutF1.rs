#pragma version(1)
#pragma rs java_package_name(ConvNetLibHS)

rs_allocation in;
int h_i, w_i;
int h_k, w_k;
int h_s, w_s;
int c, h_o, w_o;
int pad_x, pad_y;
bool kernel_type;   // false: max, true: mean

static inline float max_frame(int cur_c, int cur_h, int cur_w)
{
    int x_l = cur_h * h_s;
    int x_h = x_l + h_k;
    int y_l = cur_w * w_s;
    int y_h = y_l + w_k;

    float max;

    if (x_h > h_i + 2 * pad_x)
        x_h = h_i + 2 * pad_x;

    if (y_h > w_i + 2 * pad_y)
        y_h = w_i + 2 * pad_y;

    if (((x_l >= pad_x) && (x_l < h_i + pad_x)) || ((y_l >= pad_y) && (y_l < w_i + pad_y)))
        max = rsGetElementAt_float(in, cur_c * h_i * w_i + (x_l - pad_x) * w_i + (y_l - pad_y));
    else
        max = 0;

    for (int i = x_l; i < x_h; ++i)
        for (int j = y_l; j < y_h; ++j)
            if (i < pad_x || i >= h_i + pad_x || j < pad_y || j >= w_i + pad_y)
                max = fmax(max, 0);
            else
                max = fmax(max, rsGetElementAt_float(in, cur_c * h_i * w_i + i * w_i + j));

    return max;
}

static inline float mean_frame(int cur_c, int cur_h, int cur_w)
{
    int x_l = cur_h * h_s;
    int x_h = x_l + h_k;
    int y_l = cur_w * w_s;
    int y_h = y_l + w_k;

    float sum = 0;

    if (x_h > h_i + 2 * pad_x)
        x_h = h_i + 2 * pad_x;

    if (y_h > w_i + 2 * pad_y)
        y_h = w_i + 2 * pad_y;

    for (int i = x_l; i < x_h; ++i)
        for (int j = y_l; j < y_h; ++j)
            if (!(i < pad_x || i >= h_i + pad_x || j < pad_y || j >= w_i + pad_y))
                sum += rsGetElementAt_float(in, cur_c * h_i * w_i + i * w_i + j);

    return sum / (float) ((x_h - x_l) * (y_h - y_l));
}

float __attribute__((kernel)) root(uint32_t x)
{
        int cur_w = x % w_o;
        int cur_c = x / (h_o * w_o);
        int cur_h = (x % (h_o * w_o)) / w_o;

        float out;

        // Compute pixel value.
        if (kernel_type)
            out = mean_frame(cur_c, cur_h, cur_w);
        else
            out = max_frame(cur_c, cur_h, cur_w);

        return out;
}
