package layers;

import android.util.Log;

import numdroid.*;

public class LocalResponseNormalization implements LayerInterface {

    private String name;                        // name of the layer
    private MyNum myNum;                        // for mathematical calculations
    private int localSize;                      // local size
    private double alpha;                       // alpha
    private double beta;                        // beta
    private String normRegion;                  // norm region: "across_channels"
    private boolean parallel;                   // implementation method (parallel or sequential)

    public LocalResponseNormalization(int localSize, double alpha, double beta, String normRegion,
                                      boolean parallel, String name) {
        this.localSize = localSize;
        this.alpha = alpha;
        this.beta = beta;
        this.normRegion = normRegion;
        this.parallel = parallel;
        this.name = name;
        myNum = new MyNum();
    }

    @Override
    public Object compute(Object input) {

        Object output;

        long runTime = System.currentTimeMillis();

        if (!parallel)
            output = lrnLayerSeq((float[][][][])input, localSize, alpha, beta, normRegion);
        else
            output = lrnLayerMultithread((float[][][][])input, localSize, alpha, beta, normRegion);

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
            MultiThreadLrn[] threads = new MultiThreadLrn[n_i];
            for (int n = 0; n < n_i; ++n)
            {
                threads[n] = new MultiThreadLrn(inputBlob, n, c_i, h_i, w_i, localSize, alpha, beta, myNum);
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
        }

        return outputBlob;
    }

}

class MultiThreadLrn extends Thread {
    private float[][][][] inputBlob;
    private int n, c_i, h_i, w_i, localSize;
    private double alpha, beta;
    private MyNum myNum;

    public float[][][] outputBlob;
    public boolean done;

    public MultiThreadLrn(float[][][][] inputBlob, int n, int c_i, int h_i, int w_i, int localSize,
                          double alpha, double beta, MyNum myNum) {
        this.inputBlob = inputBlob;
        this.n = n;
        this.c_i = c_i;
        this.h_i = h_i;
        this.w_i = w_i;
        this.localSize = localSize;
        this.alpha = alpha;
        this.beta = beta;
        this.myNum = myNum;

        outputBlob = new float[c_i][h_i][w_i];
    }

    @Override
    public void run() {
        for (int c = 0; c < c_i; ++c)
            // For first few channels, do zero padding.
            if (c < (localSize - 1) / 2)
                outputBlob[c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, n, 0, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
                // For last few channels, do zero padding.
            else if (c > c_i - (localSize - 1) / 2 - 1)
                outputBlob[c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.sum(myNum.power(inputBlob, n, c - (localSize - 1) / 2, c_i, h_i, w_i, 2)), (float)alpha / localSize), 1), beta));
            else
                outputBlob[c] = myNum.divide(inputBlob[n][c], myNum.power(myNum.sum(myNum.multiply(myNum.mean(myNum.power(inputBlob, n, c - (localSize - 1) / 2, c + (localSize - 1) / 2 + 1, h_i, w_i, 2)), (float) alpha), 1), beta));

        done = true;
    }
}
