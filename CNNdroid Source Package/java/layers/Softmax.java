package layers;

import android.util.Log;

import numdroid.MyNum;

public class Softmax implements LayerInterface {
    private String name;                // name of the layer
    private MyNum myNum;                // for mathematical calculations

    public Softmax(String name) {
        this.name = name;
        myNum = new MyNum();
    }

    @Override
    public Object compute(Object input) {

        long runTime = System.currentTimeMillis();

        Object output;
        if (input.getClass().toString().equals("class [[[[F"))
            output = softmaxLayer((float[][][][])input);
        else
            output = softmaxLayer((float[][]) input);

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }

    private float[][] softmaxLayer(float[][] inputBlob) {
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;

        //initialize the result
        float[][] outputBlob = new float[n_i][c_i];

        //calculate the result
        for (int n = 0  ; n <(n_i) ; n++)
            outputBlob[n] = myNum.averaged_exp(inputBlob[n]);         //  vect = inputBlob[n].ravel()  //  vect_exp = np.exp(vect)  // outputBlob[n,:]=vect_exp / np.sum(vect_exp)

        //return the result
        return outputBlob;
    }

    private float[][] softmaxLayer(float[][][][] inputBlob) {
        int n_i, c_i;
        n_i = inputBlob.length;
        c_i = inputBlob[0].length;
        float[][] data = new float[n_i][c_i];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i ; i++)
                data[n][i] = inputBlob[n][i][0][0];

        return softmaxLayer(data);
    }

}
