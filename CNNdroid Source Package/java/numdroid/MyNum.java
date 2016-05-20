package numdroid;

public class MyNum
{
    public float sum_conv(float[][][] frames, float[][][] kernel, int x, int y, int pad_x , int pad_y)
    {
        int i_k = kernel.length;
        int h_k = kernel[0].length;
        int w_k = kernel[0][0].length;

        float sum = 0;

        for (int i = 0 ; i < i_k ; i++)
            for (int h = 0 ; h < h_k ; h++)
                for (int w = 0 ; w < w_k ; w++)
                {
                    float frame_value;
                    int cur_x = x + h;
                    int cur_y = y + w;
                    if (cur_x < pad_x || cur_x >= (pad_x + frames[0].length))
                        frame_value = 0;
                    else if (cur_y < pad_y || cur_y >= (pad_y + frames[0][0].length))
                        frame_value = 0;
                    else
                        frame_value = frames[i][cur_x - pad_x][cur_y - pad_y];

                    sum += frame_value * kernel[i][h][w];
                }

        return sum;
    }

    public float sum_innerproduct_layer4(float[][][] in1 , float[] in2 , int wIter, int w_w, int c, int h, int w)
    {
        float sum = 0;

        for (int i = 0 ; i < c ; ++i)
            for (int j = 0 ; j < h ; ++j)
                for (int k = 0 ; k < w ; ++k)
                    sum += in1[i][j][k] * in2[wIter * w_w + i * h * w + j * w + k];

        return sum;
    }

    public float sum_innerproduct_layer2(float[] in1 , float[] in2 , int wIter, int w_w, int c)
    {
        float sum = 0;

        for (int i = 0 ; i < c ; i++)
            sum += in1[i] * in2[wIter * w_w + i];
        return sum;

    }

    public float[] averaged_exp(float[] input)
    {
        int size = input.length;
        float[] output = new float[size];

        float sum = 0;
        for (int i = 0 ; i < size ; i++)
        {
            output[i] = (float) Math.pow(2.71828, (float) input[i]);
            sum += output[i];
        }
        for (int i = 0 ; i < size ; i++)
            output[i] /= sum;

        return output;
    }

    public float frame_max(float[][] frames, int x_l, int x_h, int y_l, int y_h, int h_i, int w_i, int[] pad)
    {
        float max;

        if (x_h > h_i + 2 * pad[0])
            x_h = h_i + 2 * pad[0];

        if (y_h > w_i + 2 * pad[1])
            y_h = w_i + 2 * pad[1];

        if (((x_l >= pad[0]) && (x_l < h_i + pad[0])) || (y_l >= pad[1]) && (y_l < w_i + pad[1]))
            max = frames[x_l - pad[0]][y_l - pad[1]];
        else
            max = 0;

        for (int i = x_l; i < x_h; ++i)
            for (int j = y_l; j < y_h; ++j)
                if (i < pad[0] || i >= h_i + pad[0] || j < pad[1] || j >= w_i + pad[1])
                    max = Math.max(max, 0);
                else
                    max = Math.max(max, frames[i - pad[0]][j - pad[1]]);

        return max;
    }

    public float frame_mean(float[][] frames, int x_l, int x_h, int y_l, int y_h, int h_i, int w_i, int[] pad)
    {
        float sum = 0;

        if (x_h > h_i + 2 * pad[0])
            x_h = h_i + 2 * pad[0];

        if (y_h > w_i + 2 * pad[1])
            y_h = w_i + 2 * pad[1];

        for (int i = x_l; i < x_h; ++i)
            for (int j = y_l; j < y_h; ++j)
                if (!(i < pad[0] || i >= h_i + pad[0] || j < pad[1] || j >= w_i + pad[1]))
                    sum += frames[i - pad[0]][j - pad[1]];

        return sum / (float) ((x_h - x_l) * (y_h - y_l));
    }

    public float[][][] power(float[][][][] input, int n, int c_l, int c_h, int h, int w, double p)
    {
        float[][][] output = new float[c_h - c_l][h][w];

        for (int i = c_l; i < c_h; ++i)
            for (int j = 0; j < h; ++j)
                for (int k = 0; k < w; ++k)
                    output[i - c_l][j][k] = (float) Math.pow(input[n][i][j][k], p);

        return output;
    }

    public float[][] power(float[][] input, double p)
    {
        float[][] output = new float[input.length][input[0].length];

        for (int i = 0; i < input.length; ++i)
            for (int j = 0; j < input[0].length; ++j)
                output[i][j] = (float) Math.pow(input[i][j], p);
        return output;
    }

    public float[][] sum(float[][][] input)
    {
        float[][] sum = new float[input[0].length][input[0][0].length];

        for (int i = 0; i < input[0].length; ++i)
            for (int j = 0; j < input[0][0].length; ++j)
                for (int k = 0; k < input.length; ++k)
                    sum[i][j] += input[k][i][j];

        return sum;
    }

    public float[][] sum(float[][] input, float n)
    {
        float[][] output = new float[input.length][input[0].length];

        for (int i = 0; i < input.length; ++i)
            for (int j = 0; j < input[0].length; ++j)
                output[i][j] = input[i][j] + n;

        return output;
    }

    public float[][] divide(float[][] input, float n)
    {
        float[][] output = new float[input.length][input[0].length];

        for (int i = 0; i < input.length; ++i)
            for (int j = 0; j < input[0].length; ++j)
                output[i][j] = input[i][j] / n;

        return output;
    }

    public float[][] divide(float[][] input1, float[][] input2)
    {
        if (input1.length != input2.length && input1[0].length != input2.length)
            return null;

        float[][] output = new float[input1.length][input1[0].length];
        for (int i = 0; i < input1.length; ++i)
            for (int j = 0; j < input1[0].length; ++j)
                output[i][j] = input1[i][j] / input2[i][j];

        return output;
    }

    public float[][] multiply(float[][] input, float n)
    {
        float[][] output = new float[input.length][input[0].length];

        for (int i = 0; i < input.length; ++i)
            for (int j = 0; j < input[0].length; ++j)
                output[i][j] = input[i][j] * n;

        return output;
    }

    public float[][] mean(float[][][] input)
    {
        int h = input[0].length;
        int w = input[0][0].length;

        float[][] sum = new float[h][w];

        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
            {
                for (int k = 0; k < input.length; ++k)
                    sum[i][j] += input[k][i][j];
                sum[i][j] /= input.length;
            }

        return sum;
    }

    public float mean(boolean[] input)
    {
        int w = input.length;

        float sum = 0;

        for (int i = 0; i < w; ++i)
            if (input[i])
                ++sum;

        return sum / (float) w;
    }

    public int[][] argsort(float[][] input)
    {
        int h = input.length;
        int w = input[0].length;
        int[][] index = new int[h][w];

        for (int i = 0; i < h; ++i)
            index[i] = sort(input[i]);

        return index;
    }

    public int[] sort(float[] input)
    {
        int h = input.length;
        int[] index = new int[h];

        for (int i = 0; i < h; ++i)
            index[i] = i;

        for (int i = h - 1; i > 0; --i)
        {
            // Find the index of the maximum of input[0] to input[i].
            int max = 0;
            for (int j = 1; j <= i; ++j)
                if (input[index[j]] > input[index[max]])
                    max = j;

            int temp = index[i];
            index[i] = index[max];
            index[max] = temp;
        }

        return index;
    }

    public int[][] tile_transpose(int[] input, int n)
    {
        int w = input.length;

        int[][] output = new int[w][n];

        for (int j = 0; j < n; ++j)
            for (int i = 0; i < w; ++i)
                output[i][j] = input[i];

        return output;
    }

    public float[][] dot(float[][] in1, float[][] in2)
    {
        if (in1[0].length != in2.length)
            return null;

        float[][] out = new float[in1.length][in2[0].length];

        for (int i = 0; i < in1.length; ++i)
            for (int j = 0; j < in2[0].length; ++j)
                for (int k = 0; k < in1[0].length; ++k)
                    out[i][j] += in1[i][k] * in2[k][j];

        return out;
    }

    public float[][] reshape(float[][] in, int k, int h_o, int w_o, float bias)
    {
        if (in.length != h_o * w_o)
            return null;

        float[][] out = new float[h_o][w_o];

        int n = 0;
        for (int i = 0; i < h_o; ++i)
            for (int j = 0; j < w_o; ++j)
            {
                out[i][j] = in[n][k] + bias;
                ++n;
            }

        return out;
    }

    public float squaredError(float[][][][] input1, float[][][][] input2)
    {
        float error = 0;
        for (int i = 0; i < input1.length; ++i)
            for (int j = 0; j < input1[0].length; ++j)
                for (int k = 0; k < input1[0][0].length; ++k)
                    for (int l = 0; l < input1[0][0][0].length; ++l)
                        error += (float) Math.pow((input1[i][j][k][l] - input2[i][j][k][l]) , 2);

        return (float) (error);
    }


    public float squaredError(float[][] input1, float[][][][] input2)
    {

        float error = 0;
            for (int i = 0; i < input1.length; ++i)
                for (int j = 0; j < input1[0].length; ++j)
                    error += (float) Math.pow((input1[i][j] - input2[i][j][0][0]) , 2);

        return (float) Math.sqrt(error);
    }

    public float squaredError(float[][] input1, float[][] input2)
    {

        float error = 0;
        for (int i = 0; i < input1.length; ++i)
            for (int j = 0; j < input1[0].length; ++j)
                error += (float) Math.pow((input1[i][j] - input2[i][j]) , 2);

        return (float) Math.sqrt(error);
    }
}
