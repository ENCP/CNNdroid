package layers;

import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.Scanner;

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
    private String tuningFolder;                // location to store online tuning results
    private boolean tuneNow;                    // flag to weather execute tuning ro not
    private boolean tuneFunc;                   // flag of optional tuning function
    private int threadCount;                    // thread count for acceleration
    private int[] threadCounts = {4, 6, 8};

    // types of non-linear layer that may be appended to this layer
    public enum NonLinearType {
        RectifiedLinearUnit,
        None
    }

    public Pooling(int[] kernelSize, String kernelType, int[] pad, int[] stride, boolean parallel, boolean tuneFunc, String name, String tuningFolder) {
        this.kernelSize = kernelSize;
        this.kernelType = kernelType;
        this.pad = pad;
        this.stride = stride;
        this.parallel = parallel;
        this.tuneFunc = tuneFunc;
        this.nonLinearType = NonLinearType.None;
        this.tuningFolder = tuningFolder;
        nonLinear = false;
        this.name = name;
        myNum = new MyNum();

        tuneNow = false;
        File f = new File(tuningFolder + "/" + name + ".txt");
        try {
            Scanner s = new Scanner(f);
            threadCount = Integer.valueOf(s.nextLine());
            if (corrupted(threadCount))
                tuneNow = true;
        } catch (FileNotFoundException e) {
            tuneNow = true;
        }

        if (!tuneFunc) {
            threadCount = 4;
            tuneNow = false;
        }
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
        else if (tuneNow)
            output = tuneFunction((float[][][][]) input);
        else
            output = poolLayerMultithread((float[][][][]) input, kernelSize, kernelType, pad, stride, threadCount, nonLinear);

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
                                              int threadCount, boolean hasRelu) {
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

        for (int frame = 0 ; frame < n_i ; frame++) {
            // Calculate the result
            MultiThreadPool[] threads = new MultiThreadPool[threadCount];
            for (int thread = 0; thread < threadCount ; ++thread) {
                threads[thread] = new MultiThreadPool(inputBlob, kernelSize, kernelType, pad, stride, frame, c_o, h_o, w_o, myNum, thread, threadCount, hasRelu);
                threads[thread].start();
            }

            while (true) {
                int thread;
                for (thread = 0; thread < threadCount ; thread++)
                    if (!threads[thread].done)
                        break;
                if (thread == threadCount)
                    break;
                try {
                    Thread.sleep(5);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            for (int thread = 0; thread < threadCount ; ++thread) {
                int channelCount = c_o / threadCount;
                if (c_o % threadCount != 0)
                    channelCount += 1;
                int cStart = thread * channelCount;
                int cEnd = cStart + channelCount;
                if (cStart > c_o)
                    cStart = c_o;
                if (cEnd > c_o)
                    cEnd = c_o;

                for (int c = cStart ; c < cEnd ; ++c)
                    outputBlob[frame][c]=threads[thread].outputBlob[c - cStart];
            }

        }
        return outputBlob;
    }

    /////////////////////////////////////////Tuning Function////////////////////////////////////////
    private Object tuneFunction(float[][][][] input){
        Log.d("CNNdroid", "layers." + name + ": Tuning process is starting...");
        long tuneTime = System.currentTimeMillis();

        tuneNow = false;
        long[] time = new long[threadCounts.length];
        for (int i = 0 ; i < threadCounts.length ; i++)
            time[i] = 0;
        long temp;
        int c_i = input[0].length;
        float[][][][] tuneInput = new float[1][c_i][input[0][0].length][input[0][0][0].length];
        tuneInput[0] = input[0];
        Object output = null;

        for (int i = 0; i < 4; i++) {
            for (int thread = 0 ; thread < threadCounts.length ; thread++) {
                temp = System.currentTimeMillis();
                output = poolLayerMultithread(input, kernelSize, kernelType, pad, stride, threadCounts[thread], nonLinear);
                time[thread] += System.currentTimeMillis() - temp;
            }
        }

        int min = 0;
        for (int i = 0; i < threadCounts.length ; i++)
            if (time[i] <= time[min])
                min = i;

        threadCount = threadCounts[min];

        writeFile(threadCount);
        tuneTime = System.currentTimeMillis() - tuneTime;
        Log.d("CNNdroid", "layers." + name + ": Tuning process finished in " + tuneTime + "ms.");
        return output;
    }

    ////////////////////////////////////////Local Functions/////////////////////////////////////////
    private boolean corrupted(int threadCount)
    {
        for (int i = 0 ; i < threadCounts.length ; i++)
            if (threadCount == threadCounts[i])
                return false;
        return true;
    }
    private void writeFile(int threadCount)
    {
        File f = new File(tuningFolder + "/" + name + ".txt");

        if(f.exists())
            f.delete();
        try {
            f.createNewFile();
            FileOutputStream fos = new FileOutputStream(f);
            fos.write(String.valueOf(threadCount).getBytes());
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

class MultiThreadPool extends Thread {
    private float[][][][] inputBlob;
    private int[] kernelSize;
    private String kernelType;
    private int[] pad;
    private int[] stride;
    private int frame, c_o, h_o, w_o;
    private MyNum myNum;
    private boolean hasRelu;
    private int cStart;
    private int cEnd;

    public float[][][] outputBlob;
    public boolean done;

    public MultiThreadPool(float[][][][] inputBlob, int[] kernelSize, String kernelType, int[] pad, int[] stride, int frame, int c_o, int h_o, int w_o, MyNum myNum, int threadNum, int threadCount, boolean hasRelu)
    {
        this.inputBlob = inputBlob;
        this.kernelSize = kernelSize;
        this.kernelType = kernelType;
        this.pad = pad;
        this.stride = stride;
        this.frame = frame;
        this.c_o = c_o;
        this.h_o = h_o;
        this.w_o = w_o;
        this.myNum = myNum;
        this.hasRelu = hasRelu;

        int channelCount = c_o / threadCount;
        if (c_o % threadCount != 0)
            channelCount += 1;
        cStart = threadNum * channelCount;
        cEnd = cStart + channelCount;
        if (cStart > c_o)
            cStart = c_o;
        if (cEnd > c_o)
            cEnd = c_o;

        channelCount = cEnd - cStart;

        outputBlob = new float[channelCount][h_o][w_o];
    }

    @Override
    public void run()
    {
        for (int c = cStart; c < cEnd ; ++c)
            outputBlob[c - cStart] = pool(inputBlob[frame][c], kernelType, kernelSize, pad, stride);
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