package layers;


import android.util.Log;

import messagepack.ParamUnpacker;
import numdroid.MyNum;

public class Accuracy implements LayerInterface {
    private String name;                    // name of the layer
    private int topk;                       // number of top predictions
    private String paramFilePath;           // name of the file which specifies the label of each image
    private MyNum myNum;                    // for mathematical calculations
    private ParamUnpacker paramUnpacker;    // for extracting the labels from the parameters file


    public Accuracy(int topk, String paramFilePath, String name) {
        this.topk = topk;
        this.paramFilePath = paramFilePath;
        this.myNum = new MyNum();
        this.paramUnpacker = new ParamUnpacker();
        this.name = name;
    }

    @Override
    public Object compute(Object input) {
        long loadTime = System.currentTimeMillis();

        float[][][][] labels = (float[][][][]) paramUnpacker.unpackerFunction(paramFilePath, float[][][][].class);

        loadTime = System.currentTimeMillis() - loadTime;

        long runTime = System.currentTimeMillis();

        Object output = accuracyLayer((float[][])input, labels, topk);

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid","layers." + name + ": Computation Run Time = " + String.valueOf(runTime) + ", Parameters Load Time = " + String.valueOf(loadTime));

        return output;
    }

    // Calculate top k prediction accuracy.
    private float accuracyLayer(float[][] inputMatrix, float[][][][] labelFloat4d, int topk) {
        // Convert label matrix to appropriate form.
        int w = labelFloat4d.length;
        int[] label = new int[w];
        for (int i = 0 ; i < w ; ++i)
            label[i] = (int) labelFloat4d[i][0][0][0];

        // Sort top k predictions.
        int[][] preds = myNum.argsort(inputMatrix);
        w = preds[0].length;

        // Repeat 'label' vector to 'preds' shape.
        int[][] tiled_label = myNum.tile_transpose(label, topk);

        // Compare 'preds' and 'label'.
        boolean[] accuracy = new boolean[tiled_label.length];

        for (int i = 0; i < tiled_label.length; ++i)
            for (int j = 0; j < topk; ++j)
                if (tiled_label[i][j] == preds[i][w - topk + j])
                {
                    accuracy[i] = true;
                    break;
                }

        // Return the result.
        return myNum.mean(accuracy);
    }
}
