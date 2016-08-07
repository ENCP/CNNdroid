package layers;

import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.Scanner;

import numdroid.*;

public class LocalResponseNormalization implements LayerInterface {

    private String name;                        // name of the layer
    private MyNum myNum;                        // for mathematical calculations
    private int localSize;                      // local size
    private double alpha;                       // alpha
    private double beta;                        // beta
    private String normRegion;                  // norm region: "across_channels"
    private boolean parallel;                   // implementation method (parallel or sequential)
    private String tuningFolder;                // location to store online tuning results
    private boolean tuneNow;                    // flag to weather execute tuning ro not
    private boolean tuneFunc;                   // flag of optional tuning function
    private int threadCount;                    // thread count for acceleration
    private int[] threadCounts = {4, 6, 8};

    public LocalResponseNormalization(int localSize, double alpha, double beta, String normRegion,
                                      boolean parallel, boolean tuneFunc, String name, String tuningFolder) {
        this.localSize = localSize;
        this.alpha = alpha;
        this.beta = beta;
        this.normRegion = normRegion;
        this.parallel = parallel;
        this.tuneFunc = tuneFunc;
        this.name = name;
        myNum = new MyNum();
        this.tuningFolder = tuningFolder;

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

    @Override
    public Object compute(Object input) {

        Object output;

        long runTime = System.currentTimeMillis();

        if (!parallel)
            output = lrnLayerSeq((float[][][][])input, localSize, alpha, beta, normRegion);
        else if (tuneNow)
            output = tuneFunction((float[][][][]) input);
        else
            output = lrnLayerMultithread((float[][][][]) input, localSize, alpha, beta, normRegion, threadCount);

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }


    ///////////////////////////////////////Sequential///////////////////////////////////////////////
    private float[][][][] lrnLayerSeq(float[][][][] inputBlob, int localSize, double alpha,
                                     double beta, String normRegion) {
        // Calculate sizes.
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        // Initialize the result.
        float[][][][] outputBlob = new float[n_i][c_i][h_i][w_i];

        // Calculate the result.
        if (normRegion.equals("across_channels"))
        {
            for (int n = 0; n < n_i; ++n)
                for (int c = 0; c < c_i; ++c)
                    // For first few channels, do zero padding.
                    if (c < (localSize - 1) / 2)
                        outputBlob[n][c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, n, 0, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
                        // For last few channels, do zero padding.
                    else if (c > c_i - (localSize - 1) / 2 - 1)
                        outputBlob[n][c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, n, c - (localSize - 1) / 2, c_i, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
                    else
                        outputBlob[n][c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.mean(myNum.power(inputBlob, n, c - (localSize - 1) / 2, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float) alpha), 1), beta));
        }

        return outputBlob;
    }


    ///////////////////////////////////////Multithread//////////////////////////////////////////////
    public float[][][][] lrnLayerMultithread(float[][][][] inputBlob, int localSize, double alpha,
                                             double beta, String normRegion, int threadCount) {
        // Calculate sizes.
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        // Initialize the result.
        float[][][][] outputBlob = new float[n_i][c_i][h_i][w_i];

        // Calculate the result.
        if (normRegion.equals("across_channels"))
        {
            for (int frame = 0 ; frame < n_i ; frame++) {
                // Calculate the result
                MultiThreadLrn[] threads = new MultiThreadLrn[threadCount];
                for (int thread = 0; thread < threadCount ; ++thread) {
                    threads[thread] = new MultiThreadLrn(inputBlob, frame, c_i, h_i, w_i, localSize, alpha, beta, myNum, thread, threadCount);
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
                    int channelCount = c_i / threadCount;
                    if (c_i % threadCount != 0)
                        channelCount += 1;
                    int cStart = thread * channelCount;
                    int cEnd = cStart + channelCount;
                    if (cStart > c_i)
                        cStart = c_i;
                    if (cEnd > c_i)
                        cEnd = c_i;

                    for (int c = cStart ; c < cEnd ; ++c)
                        outputBlob[frame][c]=threads[thread].outputBlob[c - cStart];
                }
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
                output = lrnLayerMultithread(input, localSize, alpha, beta, normRegion, thread);
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

class MultiThreadLrn extends Thread {
    private float[][][][] inputBlob;
    private int frame, c_i, h_i, w_i, localSize;
    private double alpha, beta;
    private MyNum myNum;
    private int cStart;
    private int cEnd;

    public float[][][] outputBlob;
    public boolean done;

    public MultiThreadLrn(float[][][][] inputBlob, int frame, int c_i, int h_i, int w_i, int localSize,
                          double alpha, double beta, MyNum myNum, int threadNum, int threadCount) {
        this.inputBlob = inputBlob;
        this.frame = frame;
        this.c_i = c_i;
        this.h_i = h_i;
        this.w_i = w_i;
        this.localSize = localSize;
        this.alpha = alpha;
        this.beta = beta;
        this.myNum = myNum;

        int channelCount = c_i / threadCount;
        if (c_i % threadCount != 0)
            channelCount += 1;
        cStart = threadNum * channelCount;
        cEnd = cStart + channelCount;
        if (cStart > c_i)
            cStart = c_i;
        if (cEnd > c_i)
            cEnd = c_i;

        channelCount = cEnd - cStart;


        outputBlob = new float[channelCount][h_i][w_i];
    }

    @Override
    public void run() {
        for (int c = cStart; c < cEnd ; ++c)
            // For first few channels, do zero padding.
            if (c < (localSize - 1) / 2)
                outputBlob[c - cStart] = myNum.divide(inputBlob[frame][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, frame, 0, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
                // For last few channels, do zero padding.
            else if (c > c_i - (localSize - 1) / 2 - 1)
                outputBlob[c - cStart] = myNum.divide(inputBlob[frame][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, frame, c - (localSize - 1) / 2, c_i, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
            else
                outputBlob[c - cStart] = myNum.divide(inputBlob[frame][c], myNum.power(myNum.sum(myNum.multiply(myNum.mean(myNum.power(inputBlob, frame, c - (localSize - 1) / 2, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float) alpha), 1), beta));

        done = true;
    }
}
