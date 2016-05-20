package layers;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.Scanner;

import messagepack.ParamUnpacker;
import numdroid.MyNum;

public class FullyConnected implements LayerInterface {
    private String name;                    // name of the layer
    private String paramFilePath;           // name of the file which specifies the weights and biases
    private ParamUnpacker paramUnpacker;    // for extracting the wieghts and biases from the parameters file
    private MyNum myNum;                    // for mathematical calculations
    private RenderScript myRS;              // RenderScript object
    private boolean nonLinear;              // Does a non-linear layer follow this layer?
    private NonLinearType nonLinearType;    // non-linearity type (if applicable)
    private boolean parallel;               // implementation method (parallel or sequential)
    private boolean loadParamsAtStart;		// if true, layer parameters will be loaded at the construction of network, otherwise the parameters will be loaded in run time
    private float[] weight; 			    // weight parameter of network
    private float[] bias;					// bias parameter of network
    private String tuningFolder;            // location to store online tuning results
    private boolean tune;                   // flag to weather execute tuning ro not
    private String algorithm;               // acceleration method
    private String[] names = {"F4F1", "F8F1"};

    private ScriptC_innerProductInF4OutF1 myScriptF4;
    private ScriptC_innerProductInF8OutF1 myScriptF8;


    // types of non-linear layer that may be appended to this layer
    public enum NonLinearType {
        RectifiedLinearUnit,
        None
    }

    public FullyConnected(String paramFilePath, boolean parallel, boolean loadParamsAtStart, RenderScript myRS, String name, String tuningFolder) {
        this.paramFilePath = paramFilePath;
        this.parallel = parallel;
        this.myRS = myRS;
        this.nonLinearType = NonLinearType.None;
        this.nonLinear = false;
        this.name = name;
        this.myNum = new MyNum();
        this.paramUnpacker = new ParamUnpacker();
        this.loadParamsAtStart = loadParamsAtStart;
        this.tuningFolder = tuningFolder;

        tune = false;
        File f = new File(tuningFolder + "/" + name + ".txt");
        try {
            Scanner s = new Scanner(f);
            algorithm = s.nextLine();
            if (corrupted(algorithm))
                tune = true;
        } catch (FileNotFoundException e) {
            tune = true;
        }
		  
       if (loadParamsAtStart && (!tune || !parallel)) {
          long loadTime = System.currentTimeMillis();
          Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[].class, float[].class});
          weight = (float[]) objects[0];
          bias = (float[]) objects[1];

          loadTime = System.currentTimeMillis() - loadTime;
          Log.d("CNNdroid","layers." + name + ": Parameters Load Time in Constructor = " + String.valueOf(loadTime) + ", Shape: " + bias.length);

          if (parallel)
          {
              long kernelTime = System.currentTimeMillis();
              switch (algorithm) {
                  case "F4F1":
                      initKernelF4F1(weight, bias);
                      break;
                  case "F8F1":
                      initKernelF8F1(weight, bias);
                      break;
              }
              kernelTime = System.currentTimeMillis() - kernelTime;
              Log.d("CNNdroid", "layers." + name + ": Kernel Initialization Time in Constructor = " + String.valueOf(kernelTime));
          }
       }
    }

    public void setNonLinearType(NonLinearType nonLinearType) {
        this.nonLinearType = nonLinearType;
        nonLinear = true;
    }

    @Override
    public Object compute(Object input) {
        long loadTime;
        if (!loadParamsAtStart && (!tune || !parallel)) {
		    loadTime = System.currentTimeMillis();

		    Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[].class, float[].class});
		    float[] localWeight = (float[]) objects[0];
		    float[] localBias = (float[]) objects[1];

            if (parallel) {
                switch (algorithm) {
                    case "F4F1":
                        initKernelF4F1(localWeight, localBias);
                        break;
                    case "F8F1":
                        initKernelF8F1(localWeight, localBias);
                        break;
                }
            }
		    loadTime = System.currentTimeMillis() - loadTime;
            Log.d("CNNdroid","layers." + name + ": Parameters Load Time = " + String.valueOf(loadTime));

            return invokeFunctions(input, localWeight, localBias, true);
        }

        else
        {
            return invokeFunctions(input,weight, bias, false);
        }
    }

    ///////////////////////////////////////Sequential///////////////////////////////////////////////
    private float[][] fullyConnectedLayerSeq(float[][] inputBlob2, float[] weight, float[] bias) {
        // fully connected layer
        int h_w = bias.length;
        int w_w = weight.length / h_w;

        // Calculate sizes.
        int n_i, c_i;

        n_i = inputBlob2.length;
        c_i = inputBlob2[0].length;

        int n_o = n_i;
        int c_o = h_w;

        // Initialize the result.
        float[][] outputBlob = new float[n_o][c_o];

        // Calculate inner product.
        for (int n = 0; n < n_i; n++)
            for (int c = 0; c < c_o; c++)
                outputBlob[n][c] = myNum.sum_innerproduct_layer2(inputBlob2[n], weight, c, w_w, c_i) + bias[c];

        // return the result
        return outputBlob;
    }

    private float[][] fullyConnectedLayerSeq(float[][][][] inputBlob4, float[] weight, float[] bias) {
        // fully connected layer
        int h_w = bias.length;
        int w_w = weight.length / h_w;

        // Calculate sizes.
        int n_i, c_i, h_i, w_i;
        n_i = inputBlob4.length;
        c_i = inputBlob4[0].length;
        h_i = inputBlob4[0][0].length;
        w_i = inputBlob4[0][0][0].length;

        int n_o = n_i;
        int c_o = h_w;

        // Initialize the result.
        float[][] outputBlob = new float[n_o][c_o];

        // Calculate inner product.
        for (int n = 0; n < n_i; n++)
            for (int c = 0; c < c_o; c++)
                outputBlob[n][c] = myNum.sum_innerproduct_layer4(inputBlob4[n], weight, c, w_w, c_i, h_i, w_i) + bias[c];

        // return the result
        return outputBlob;
    }

    ////////////////////////////////////////Parallel////////////////////////////////////////////////
    // Input: Float4     *****   Output: Float
    private float[][] fullyConnectedLayerInF4OutF1(float[][] inputBlob4, float[] myWeight, float[] myBias, boolean destroy) {
        // fully connected layer

        int h_w = myBias.length;
        int w_w = myWeight.length / h_w;

        // Calculate sizes.
        int n_i, c_i;
        n_i = inputBlob4.length;
        c_i = inputBlob4[0].length;

        int c_i_4 = c_i;
        if (c_i % 4 != 0)
            c_i_4 = c_i + 4 - c_i % 4;

        int n_o = n_i;
        int c_o = h_w;

        // Initialize the result.
        float[][] outputBlob = new float[n_o][c_o];

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;
        inputType = Type.createX(myRS, Element.F32_4(myRS), n_i * c_i_4 / 4);
        outType = Type.createX(myRS, Element.F32(myRS), n_o * c_o);

        frameAllocation = Allocation.createTyped(myRS, inputType);
        outAllocation = Allocation.createTyped(myRS, outType);

        myScriptF4.set_c_i(c_i_4);
        // calculate the result

        float[] frameMatrix = new float[n_i * c_i_4];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i_4 ; i++)
                if (i < c_i_4)
                    frameMatrix[n * w_w + i] = inputBlob4[n][i];
                else
                    frameMatrix[n * w_w + i] = 0;

        frameAllocation.copyFrom(frameMatrix);
        myScriptF4.set_In_Blob(frameAllocation);


        myScriptF4.forEach_root(outAllocation);

        float[] outMatrix = new float[n_o * c_o];
        outAllocation.copyTo(outMatrix);

        for (int n = 0 ; n < n_i ; n++)
            for (int c = 0 ; c < c_o ; c++) {
                outputBlob[n][c] = outMatrix[n * c_o + c];
                if (nonLinear) {
                    switch (nonLinearType) {
                        case RectifiedLinearUnit:
                            if (outputBlob[n][c] < 0)
                                outputBlob[n][c] = 0;
                            break;
                    }
                }
            }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy) {
            myScriptF4.destroy();
            myScriptF4 = null;
        }

        // return the result
        return outputBlob;
    }

    // Input: Float4     *****   Output: Float
    private float[][] fullyConnectedLayerInF4OutF1(float[][][][] inputBlob4, float[] myWeight, float[] myBias, boolean destroy) {
        int n_i, c_i, h_i = 0, w_i = 0;
        n_i = inputBlob4.length;
        c_i = inputBlob4[0].length;
        h_i = inputBlob4[0][0].length;
        w_i = inputBlob4[0][0][0].length;
        float[][] data = new float[n_i][c_i * h_i * w_i];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i ; i++)
                for (int j = 0 ; j < h_i ; j++)
                    for (int k = 0 ; k < w_i ; k++)
                        data[n][i * h_i * w_i + j * w_i + k] = inputBlob4[n][i][j][k];

        return fullyConnectedLayerInF4OutF1(data, myWeight, myBias, destroy);
    }

    // Input: Float8     *****   Output: Float
    private float[][] fullyConnectedLayerInF8OutF1(float[][] inputBlob4, float[] myWeight, float[] myBias, boolean destroy) {
        // fully connected layer

        int h_w = myBias.length;
        int w_w = myWeight.length / h_w;

        // Calculate sizes.
        int n_i, c_i;
        n_i = inputBlob4.length;
        c_i = inputBlob4[0].length;

        int c_i_8 = c_i;
        if (c_i % 8 != 0)
            c_i_8 = c_i + 8 - c_i % 8;

        int n_o = n_i;
        int c_o = h_w;

        // Initialize the result.
        float[][] outputBlob = new float[n_o][c_o];

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;
        inputType = Type.createX(myRS, Element.F32_4(myRS), n_i * c_i_8 / 4);
        outType = Type.createX(myRS, Element.F32(myRS), n_o * c_o);

        frameAllocation = Allocation.createTyped(myRS, inputType);
        outAllocation = Allocation.createTyped(myRS, outType);

        myScriptF8.set_c_i(c_i_8);
        // calculate the result

        float[] frameMatrix = new float[n_i * c_i_8];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i_8 ; i++)
                if (i < c_i_8)
                    frameMatrix[n * w_w + i] = inputBlob4[n][i];
                else
                    frameMatrix[n * w_w + i] = 0;

        frameAllocation.copyFrom(frameMatrix);
        myScriptF8.set_In_Blob(frameAllocation);


        myScriptF8.forEach_root(outAllocation);

        float[] outMatrix = new float[n_o * c_o];
        outAllocation.copyTo(outMatrix);

        for (int n = 0 ; n < n_i ; n++)
            for (int c = 0 ; c < c_o ; c++) {
                outputBlob[n][c] = outMatrix[n * c_o + c];
                if (nonLinear) {
                    switch (nonLinearType) {
                        case RectifiedLinearUnit:
                            if (outputBlob[n][c] < 0)
                                outputBlob[n][c] = 0;
                            break;
                    }
                }
            }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy) {
            myScriptF8.destroy();
            myScriptF8 = null;
        }

        // return the result
        return outputBlob;
    }

    // Input: Float8     *****   Output: Float
    private float[][] fullyConnectedLayerInF8OutF1(float[][][][] inputBlob4, float[] myWeight, float[] myBias, boolean destroy) {
        int n_i, c_i, h_i, w_i;
        n_i = inputBlob4.length;
        c_i = inputBlob4[0].length;
        h_i = inputBlob4[0][0].length;
        w_i = inputBlob4[0][0][0].length;
        float[][] data = new float[n_i][c_i * h_i * w_i];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i ; i++)
                for (int j = 0 ; j < h_i ; j++)
                    for (int k = 0 ; k < w_i ; k++)
                        data[n][i * h_i * w_i + j * w_i + k] = inputBlob4[n][i][j][k];

        return fullyConnectedLayerInF8OutF1(data, myWeight, myBias, destroy);
    }


    ///////////////////////////////Kernel Initialization Functions//////////////////////////////////
    void initKernelF4F1(float[] myWeight, float[] myBias)
    {
        int h_w = myBias.length;
        int w_w = myWeight.length / h_w;

        Type kernelType,biasType;
        Allocation kernelAllocation;
        Allocation biasAllocation;
        kernelType = Type.createX(myRS, Element.F32_4(myRS), h_w * w_w / 4);
        biasType = Type.createX(myRS, Element.F32(myRS), h_w);


        kernelAllocation = Allocation.createTyped(myRS, kernelType);
        kernelAllocation.copyFrom(myWeight);

        biasAllocation = Allocation.createTyped(myRS, biasType);
        biasAllocation.copyFrom(myBias);

        myScriptF4 = new ScriptC_innerProductInF4OutF1(myRS);


        myScriptF4.set_Bias_Blob(biasAllocation);
        myScriptF4.set_Kernel_Blob(kernelAllocation);
        myScriptF4.set_w_w(w_w);
        myScriptF4.set_c_o(h_w);
    }

    void initKernelF8F1(float[] myWeight, float[] myBias)
    {
        int h_w = myBias.length;
        int w_w = myWeight.length / h_w;

        Type kernelType,biasType;
        Allocation kernelAllocation;
        Allocation biasAllocation;
        kernelType = Type.createX(myRS, Element.F32_4(myRS), h_w * w_w / 4);
        biasType = Type.createX(myRS, Element.F32(myRS), h_w);


        kernelAllocation = Allocation.createTyped(myRS, kernelType);
        kernelAllocation.copyFrom(myWeight);

        biasAllocation = Allocation.createTyped(myRS, biasType);
        biasAllocation.copyFrom(myBias);

        myScriptF8 = new ScriptC_innerProductInF8OutF1(myRS);


        myScriptF8.set_Bias_Blob(biasAllocation);
        myScriptF8.set_Kernel_Blob(kernelAllocation);
        myScriptF8.set_w_w(w_w);
        myScriptF8.set_c_o(h_w);
    }

    /////////////////////////////////////////Tuning Function////////////////////////////////////////
    private Object tuneFunction(float[][] input) {
        long tuneTime = System.currentTimeMillis();
        Log.d("CNNdroid", "layers." + name + ": Tuning process is starting...");
        Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[].class, float[].class});
        float[] myWeight = (float[]) objects[0];
        float[] myBias = (float[]) objects[1];
        tune = false;
        long[] time = new long[]{0, 0};
        long temp;

        Object output = null;

        for (int i = 0; i < 4; i++) {
            temp = System.currentTimeMillis();
            initKernelF4F1(myWeight, myBias);
            fullyConnectedLayerInF4OutF1(input, myWeight, myBias, true);
            time[0] += System.currentTimeMillis() - temp;

            temp = System.currentTimeMillis();
            initKernelF8F1(myWeight, myBias);
            output = fullyConnectedLayerInF8OutF1(input, myWeight, myBias, true);
            time[1] += System.currentTimeMillis() - temp;
        }

        int min = 0;
        for (int i = 0; i < 2; i++)
            if (time[i] <= time[min])
                min = i;

        algorithm = names[min];

        writeFile(algorithm);
        if(loadParamsAtStart) {
            weight = myWeight;
            bias = myBias;
            switch (algorithm) {
                case "F4F1":
                    initKernelF4F1(weight, bias);
                    break;
                case "F8F1":
                    initKernelF8F1(weight, bias);
                    break;
            }
        }

        tuneTime = System.currentTimeMillis() - tuneTime;
        Log.d("CNNdroid", "layers." + name + ": Tuning process finished in " + tuneTime + "mS!");
        return output;
    }

    private Object tuneFunction(float[][][][] input) {
        int n_i, c_i, h_i, w_i;
        n_i = input.length;
        c_i = input[0].length;
        h_i = input[0][0].length;
        w_i = input[0][0][0].length;
        float[][] data = new float[n_i][c_i * h_i * w_i];
        for (int n = 0 ; n < n_i ; n++)
            for (int i = 0 ; i < c_i ; i++)
                for (int j = 0 ; j < h_i ; j++)
                    for (int k = 0 ; k < w_i ; k++)
                        data[n][i * h_i * w_i + j * w_i + k] = input[n][i][j][k];

        return tuneFunction(data);
    }

    ////////////////////////////////////////Local Functions/////////////////////////////////////////
    private Object invokeFunctions(Object input, float[] myWeight, float[] myBias, boolean destroy)
    {
        Object output = null;
        long runTime = System.currentTimeMillis();

        if (!parallel) {
            if (input.getClass().toString().equals("class [[[[F"))
                output = fullyConnectedLayerSeq(((float[][][][]) input), myWeight, myBias);
            else
                output = fullyConnectedLayerSeq((float[][]) input, myWeight, myBias);
        }
        else {
            if (tune) {
                if (input.getClass().toString().equals("class [[[[F"))
                    output = tuneFunction((float[][][][]) input);
                else
                    output = tuneFunction((float[][]) input);
            }
            else {
                switch (algorithm) {
                    case "F4F1":
                        if (input.getClass().toString().equals("class [[[[F"))
                            output = fullyConnectedLayerInF4OutF1(((float[][][][]) input), myWeight, myBias, destroy);
                        else
                            output = fullyConnectedLayerInF4OutF1((float[][]) input, myWeight, myBias, destroy);
                        break;
                    case "F8F1":
                        if (input.getClass().toString().equals("class [[[[F"))
                            output = fullyConnectedLayerInF8OutF1(((float[][][][]) input), myWeight, myBias, destroy);
                        else
                            output = fullyConnectedLayerInF8OutF1((float[][]) input, myWeight, myBias, destroy);
                        break;
                }
            }
        }

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }
    private boolean corrupted(String str)
    {
        for (int i = 0 ; i < names.length ; i++)
            if (str.equals(names[i]))
                return false;
        return true;
    }
    private void writeFile(String str)
    {
        File f = new File(tuningFolder + "/" + name + ".txt");

        if(f.exists())
            f.delete();
        try {
            f.createNewFile();
            FileOutputStream fos = new FileOutputStream(f);
            fos.write(str.getBytes());
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
