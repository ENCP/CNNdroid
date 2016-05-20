package layers;

import android.util.Log;

import numdroid.MyNum;

public class Pooling implements LayerInterface {
    public final String type = "Pooling";
    private String name;                        // name of the layer
    private int[] kernelSize;                   // kernel size
    private String kernelType;                  // kernel type
    private int[] pad;                          // pad
    private int[] stride;                       // stride
    private MyNum myNum;                        // for mathematical calculations
    private boolean nonLinear;                  // Does a non-linear layer follow this layer?
    private NonLinearType nonLinearType;        // non-linearity type (if applicable)
    private boolean parallel;                   // implementation method (parallel or sequential)

    // types of non-linear layer that may be appended to this layer
    public enum NonLinearType {
        RectifiedLinearUnit,
        None
    }

    public Pooling(int[] kernelSize, String kernelType, int[] pad, int[] stride, boolean parallel, String name) {
        this.kernelSize = kernelSize;
        this.kernelType = kernelType;
        this.pad = pad;
        this.stride = stride;
        this.parallel = parallel;
        this.nonLinearType = NonLinearType.None;
        nonLinear = false;
        this.name = name;
        myNum = new MyNum();
    }

    public void setNonLinearType(NonLinearType nonLinearType) {
        this.nonLinearType = nonLinearType;
        nonLinear = true;
    }

    @Override
    public Object compute(Object input) {

        Object output;

        long runTime = System.currentTimeMillis();

        if(!parallel)
            output = poolLayerSeq((float[][][][])input, kernelSize, kernelType, pad, stride);
        else
            output = poolLayerMultithread((float[][][][])input, kernelSize, kernelType, pad, stride, nonLinear);

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }

    ///////////////////////////////////////Sequential///////////////////////////////////////////////
    private float[][][][] poolLayerSeq(float[][][][] inputBlob, int[] kernelSize,
                                      String kernelType, int[] pad, int[] stride) {
        // Calculate sizes.
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        int h_k = kernelSize[0];
        int w_k = kernelSize[1];

        int n_o = n_i;
        int c_o = c_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / (double) stride[0]) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / (double) stride[1]) + 1);

        // Initialize the result.
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];


        // Calculate the result
        for (int n = 0; n < n_i; ++n)
            for (int c = 0; c < c_i; ++c)
                outputBlob[n][c] = pool(inputBlob[n][c], kernelType, kernelSize, pad, stride);

        return outputBlob;
    }

    private float[][] pool(float[][] frames, String kernelType, int[] kernelSize,
                           int[] pad, int[] stride) {
        // Calculate final dimensions.

        int h_i = frames.length;
        int w_i = frames[0].length;

        int h_k = kernelSize[0];
        int w_k = kernelSize[1];

        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / (double) stride[0]) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / (double) stride[1]) + 1);

        int h_s = stride[0];
        int w_s = stride[1];

        // Compute pixel values.
        float[][] out = new float[h_o][w_o];


        if (kernelType.equals("max"))
            for (int x = 0; x < h_o; ++x)
                for (int y = 0; y < w_o; ++y)
                    out[x][y] = myNum.frame_max(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);
        else if (kernelType.equals("ave"))
            for (int x = 0; x < h_o; ++x)
                for (int y = 0; y < w_o; ++y)
                    out[x][y] = myNum.frame_mean(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);

        return out;
    }

    ///////////////////////////////////////Multithread//////////////////////////////////////////////
    public float[][][][] poolLayerMultithread(float[][][][] inputBlob, int[] kernelSize,
                                              String kernelType, int[] pad, int[] stride,
                                              boolean hasRelu) {
        // Calculate sizes.
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        int h_k = kernelSize[0];
        int w_k = kernelSize[1];

        int n_o = n_i;
        int c_o = c_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / (double) stride[0]) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / (double) stride[1]) + 1);

        // Initialize the result.
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];


        // Calculate the result
        MultiThreadPool[] threads = new MultiThreadPool[n_i];
        for (int n = 0; n < n_i; ++n)
        {
            threads[n] = new MultiThreadPool(inputBlob, kernelSize, kernelType, pad, stride, n, c_o, h_o, w_o, myNum, hasRelu);
            threads[n].start();
        }


        while (true)
        {
            int n;
            for (n = 0; n < n_i; ++n)
                if (!threads[n].done)
                    break;
            if (n == n_i)
                break;
            try
            {
                Thread.sleep(10);
            }
            catch (InterruptedException e)
            {
                e.printStackTrace();
            }
        }

        for (int n = 0; n < n_i; ++n)
            outputBlob[n] = threads[n].outputBlob;

        return outputBlob;
    }
}

class MultiThreadPool extends Thread {
    private float[][][][] inputBlob;
    private int[] kernelSize;
    private String kernelType;
    private int[] pad;
    private int[] stride;
    private int n, c_o, h_o, w_o;
    private MyNum myNum;
    private boolean hasRelu;

    public float[][][] outputBlob;
    public boolean done;

    public MultiThreadPool(float[][][][] inputBlob, int[] kernelSize, String kernelType, int[] pad, int[] stride, int n, int c_o, int h_o, int w_o, MyNum myNum, boolean hasRelu)
    {
        this.inputBlob = inputBlob;
        this.kernelSize = kernelSize;
        this.kernelType = kernelType;
        this.pad = pad;
        this.stride = stride;
        this.n = n;
        this.c_o = c_o;
        this.h_o = h_o;
        this.w_o = w_o;
        this.myNum = myNum;
        this.hasRelu = hasRelu;

        outputBlob = new float[c_o][h_o][w_o];
    }

    @Override
    public void run()
    {
        for (int c = 0; c < c_o; ++c)
            outputBlob[c] = pool(inputBlob[n][c], kernelType, kernelSize, pad, stride);

        done = true;
    }

    // frame pooling
    private float[][] pool(float[][] frames, String kernelType, int[] kernelSize, int[] pad, int[] stride)
    {
        // Calculate final dimensions.
        int h_i = frames.length;
        int w_i = frames[0].length;

        int h_k = kernelSize[0];
        int w_k = kernelSize[1];

        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / (double) stride[0]) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / (double) stride[1]) + 1);

        int h_s = stride[0];
        int w_s = stride[1];

        // Compute pixel values.
        float[][] out = new float[h_o][w_o];

        if (hasRelu)
        {
            if (kernelType.equals("max"))
                for (int x = 0; x < h_o; ++x)
                    for (int y = 0; y < w_o; ++y)
                        out[x][y] = myNum.frame_max(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);
            else if (kernelType.equals("ave"))
                for (int x = 0; x < h_o; ++x)
                    for (int y = 0; y < w_o; ++y)
                        out[x][y] = myNum.frame_mean(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);
        }
        else
        {
            if (kernelType.equals("max"))
                for (int x = 0; x < h_o; ++x)
                    for (int y = 0; y < w_o; ++y)
                    {
                        float f = myNum.frame_max(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);
                        if (f > 0)
                            out[x][y] = f;
                        else
                            out[x][y] = 0;
                    }
            else if (kernelType.equals("ave"))
                for (int x = 0; x < h_o; ++x)
                    for (int y = 0; y < w_o; ++y)
                    {
                        float f = myNum.frame_mean(frames, x * h_s, x * h_s + h_k, y * w_s, y * w_s + w_k, h_i, w_i, pad);
                        if (f > 0)
                            out[x][y] = f;
                        else
                            out[x][y] = 0;
                    }
        }

        return out;
    }
}
