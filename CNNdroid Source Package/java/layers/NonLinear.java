package layers;


import android.util.Log;

public class NonLinear implements LayerInterface {

    private String name;                    // name of the layer
    private NonLinearType nonLinearType;    // non-linearity type

    // types of non-linearity
    public enum NonLinearType {
        RectifiedLinearUnit
    }

    public NonLinear(String name, NonLinearType nonLinearType) {
        this.name = name;
        this.nonLinearType = nonLinearType;
    }

    @Override
    public Object compute(Object input) {
        Object output = new Object();

        long runTime = System.currentTimeMillis();

        switch (nonLinearType) {
            case RectifiedLinearUnit:
                if (input.getClass().toString().equals("class [[[[F"))
                    output =  reluLayer((float[][][][]) input);
                else
                    output =  reluLayer((float[][]) input);
                break;
        }

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }

    private float[][][][] reluLayer(float [][][][] inputBlob) {
        int n = inputBlob.length;
        int c = inputBlob[0].length;
        int h = inputBlob[0][0].length;
        int w = inputBlob[0][0][0].length;

        float[][][][] outputBlob = new float[n][c][h][w];


        for (int i = 0; i < n; ++i)
            for (int j = 0; j < c; ++j)
                for (int k = 0; k < h; ++k)
                    for (int l = 0; l < w; ++l)
                        if (inputBlob[i][j][k][l] > 0)
                            outputBlob[i][j][k][l] = inputBlob[i][j][k][l];
                        else
                            outputBlob[i][j][k][l] = 0;

        return outputBlob;
    }

    private float[][] reluLayer(float [][] inputBlob) {
        int n = inputBlob.length;
        int c = inputBlob[0].length;

        float[][] outputBlob = new float[n][c];


        for (int i = 0; i < n; ++i)
            for (int j = 0; j < c; ++j)
                if (inputBlob[i][j] > 0)
                    outputBlob[i][j] = inputBlob[i][j];
                else
                    outputBlob[i][j] = 0;

        return outputBlob;
    }
}
