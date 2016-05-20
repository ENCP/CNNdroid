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

public class Convolution implements LayerInterface {

    private String name;                    // name of the layer
    private String paramFilePath;           // name of the file which specifies the weights and biases
    private ParamUnpacker paramUnpacker;    // for extracting the wieghts and biases from the parameters file
    private int[] stride;                   // strides
    private int[] pad;                      // pads
    private int group;                      // number of groups
    private MyNum myNum;                    // for mathematical calculations
    private RenderScript myRS;              // RenderScript object
    private boolean nonLinear;              // Does a non-linear layer follow this layer?
    private NonLinearType nonLinearType;    // non-linearity type (if applicable)
    private boolean parallel;               // implementation method (parallel or sequential)
    private boolean loadParamsAtStart;        // if true, layer parameters will be loaded at the construction of network, otherwise the parameters will be loaded in run time
    private float[][][][] weight;            // weight parameter of network
    private float[] bias;                    // bias parameter of network
    private String tuningFolder;            // location to store online tuning results
    private boolean tune;                   // flag to weather execute tuning ro not
    private String algorithm;               // acceleration method
    private String[] names = {"F4F1", "F4F2", "F4F4", "F4F8", "F8F1", "F8F2", "F8F4", "F8F8"};

    private ScriptC_convRolledInF4OutF1 myScript41;
    private ScriptC_convRolledInF4OutF2 myScript42;
    private ScriptC_convRolledInF4OutF4 myScript44;
    private ScriptC_convRolledInF4OutF8 myScript48;
    private ScriptC_convRolledInF8OutF1 myScript81;
    private ScriptC_convRolledInF8OutF2 myScript82;
    private ScriptC_convRolledInF8OutF4 myScript84;
    private ScriptC_convRolledInF8OutF8 myScript88;


    // types of non-linear layer that may be appended to this layer
    public enum NonLinearType {
        RectifiedLinearUnit,
        None
    }

    public Convolution(int[] stride, int[] pad, int group, String paramFilePath, boolean parallel, boolean loadParamsAtStart, RenderScript myRS, String name, String tuningFolder) {
        this.paramFilePath = paramFilePath;
        this.stride = stride;
        this.pad = pad;
        this.group = group;
        this.nonLinearType = NonLinearType.None;
        this.nonLinear = false;
        this.myRS = myRS;
        this.parallel = parallel;
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

            Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
            weight = (float[][][][]) objects[0];
            bias = (float[]) objects[1];

            loadTime = System.currentTimeMillis() - loadTime;

            long kernelTime = System.currentTimeMillis();
            Log.d("CNNdroid", "layers." + name + ": Parameters Load Time in Constructor = " + String.valueOf(loadTime));

            if (parallel) {
                switch (algorithm) {
                    case "F4F1":
                        initKernelF4F1(weight, bias);
                        break;
                    case "F4F2":
                        initKernelF4F2(weight, bias);
                        break;
                    case "F4F4":
                        initKernelF4F4(weight, bias);
                        break;
                    case "F4F8":
                        initKernelF4F8(weight, bias);
                        break;
                    case "F8F1":
                        initKernelF8F1(weight, bias);
                        break;
                    case "F8F2":
                        initKernelF8F2(weight, bias);
                        break;
                    case "F8F4":
                        initKernelF8F4(weight, bias);
                        break;
                    case "F8F8":
                        initKernelF8F8(weight, bias);
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

            Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
            float[][][][] localWeight = (float[][][][]) objects[0];
            float[] localBias = (float[]) objects[1];

            if (parallel){
                switch (algorithm) {
                    case "F4F1":
                        initKernelF4F1(localWeight, localBias);
                        break;
                    case "F4F2":
                        initKernelF4F2(localWeight, localBias);
                        break;
                    case "F4F4":
                        initKernelF4F4(localWeight, localBias);
                        break;
                    case "F4F8":
                        initKernelF4F8(localWeight, localBias);
                        break;
                    case "F8F1":
                        initKernelF8F1(localWeight, localBias);
                        break;
                    case "F8F2":
                        initKernelF8F2(localWeight, localBias);
                        break;
                    case "F8F4":
                        initKernelF8F4(localWeight, localBias);
                        break;
                    case "F8F8":
                        initKernelF8F8(localWeight, localBias);
                        break;
                }
            }
            loadTime = System.currentTimeMillis() - loadTime;
            Log.d("CNNdroid", "layers." + name + ": Parameters Load Time = " + String.valueOf(loadTime));

            return invokeFunctions(input, localWeight, localBias, true);
        }
        else
        {
            return invokeFunctions(input,weight, bias, false);
        }
    }


    ///////////////////////////////////////Sequential///////////////////////////////////////////////
    private float[][][][] convLayerRolledSeq(float[][][][] inputBlob, float[][][][] filterBlob,
                                             float[] biasBlob, int[] pad, int[] stride, int group) {
        /*
        Convolution Layer
        Inputs:
        kernel[0] is a filter blob.
        kernel[1] is bias blob.
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = filterBlob.length;
        int c_k = filterBlob[0].length;
        int h_k = filterBlob[0][0].length;
        int w_k = filterBlob[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        // calculate the result
        for (int n = 0; n < (n_i); n++) // for n in images
            for (int k = 0; k < (n_k / group); k++)// for k in kernels
                for (int g = 0; g < (group); g++) {
                    float[][][] convInFrame = new float[(c_i / group)][h_i][w_i];
                    float[][][] convInKernel = new float[(c_i / group)][h_i][w_i];

                    int temp = g * c_i / group;
                    for (int i = g * c_i / group; i < (g + 1) * c_i / group; i++) // copy part of inputBlob
                        convInFrame[i - temp] = inputBlob[n][i];

                    convInKernel = filterBlob[g * n_k / group + k];       // copy

                    outputBlob[n][k + g * n_k / group] = convRolledSeq(convInFrame, convInKernel, biasBlob[g * n_k / group + k], pad, stride);
                }

        // return the result
        return outputBlob;
    }

    private float[][] convRolledSeq(float[][][] frames, float[][][] kernel, float bias,
                                    int[] pad, int[] stride) {
        // Calculate final dimensions.
        int c_i = frames.length;
        int h_i = frames[0].length;
        int w_i = frames[0][0].length;

        int c_k = kernel.length;
        int h_k = kernel[0].length;
        int w_k = kernel[0][0].length;

        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) stride[0])) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) stride[1])) + 1);

        int h_s = stride[0];
        int w_s = stride[1];

        float[][] out = new float[h_o][w_o];

        // Compute pixel values.
        for (int i = 0; i < h_o; ++i)
            for (int j = 0; j < w_o; ++j)
                out[i][j] = myNum.sum_conv(frames, kernel, i * h_s, j * w_s, pad[0], pad[1]) + bias;

        return out;
    }


    ////////////////////////////////////////Parallel////////////////////////////////////////////////
    // Input: Float4     *****   Output: Float
    private float[][][][] convLayerRolledParInF4OutF1(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_4 = c_i;
        if (c_i % (4 * group) != 0)
            c_i_4 = c_i + (4 * group) - c_i % (4 * group);

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_4 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32(myRS), h_o * w_o * n_k);


        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        myScript41.set_c_i(c_i_4);
        myScript41.set_h_i(h_i);
        myScript41.set_w_i(w_i);
        myScript41.set_h_o(h_o);
        myScript41.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k];
        float[] frameMatrix = new float[h_i * w_i * c_i_4];
        int delta_c = (c_i_4 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript41.set_In_Blob(frameAllocation);

            myScript41.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            outputBlob[n - 1][i][j][k] = outMatrix[i * h_o * w_o + j * w_o + k];
                            if (nonLinear)
                                if (outputBlob[n - 1][i][j][k] < 0)
                                    outputBlob[n - 1][i][j][k] = 0;
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            outputBlob[n][i][j][k] = outMatrix[i * h_o * w_o + j * w_o + k];
                            if (nonLinear)
                                if (outputBlob[n][i][j][k] < 0)
                                    outputBlob[n][i][j][k] = 0;
                        }
            }

        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript41.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float4    *****   Output: Float2
    private float[][][][] convLayerRolledParInF4OutF2(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_4 = c_i;
        if (c_i % (4 * group) != 0)
            c_i_4 = c_i + (4 * group) - c_i % (4 * group);

        int n_k_2 = n_k;
        if (n_k % (2 * group) != 0)
            n_k_2 = n_k + (2 * group) - n_k % (2 * group);

        int delta_n = (n_k_2 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_4 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_2(myRS), h_o * w_o * n_k_2 / 2);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        myScript42.set_c_i(c_i_4);
        myScript42.set_h_i(h_i);
        myScript42.set_w_i(w_i);
        myScript42.set_h_o(h_o);
        myScript42.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k_2];
        float[] frameMatrix = new float[h_i * w_i * c_i_4];
        int delta_c = (c_i_4 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript42.set_In_Blob(frameAllocation);

            myScript42.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_2 / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_2 / group) && (i < n_k_2 - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_2 / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_2 / group) && (i < n_k_2 - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript42.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float4    *****   Output: Float4
    private float[][][][] convLayerRolledParInF4OutF4(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        //check channel count
        int c_k_4 = c_k;
        if (c_k % 4 != 0)
            c_k_4 = c_k + 4 - c_k % 4;

        int c_i_4 = c_i;
        if (c_i % 4 != 0)
            c_i_4 = c_i + 4 - c_i % 4;

        int n_k_4 = n_k;
        if (n_k % 4 != 0)
            n_k_4 = n_k + 4 - n_k % 4;

        int delta_n = (n_k_4 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_4 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_4(myRS), h_o * w_o * n_k_4 / 4);


        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);


        myScript44.set_c_i(c_i_4);
        myScript44.set_h_i(h_i);
        myScript44.set_w_i(w_i);
        myScript44.set_h_o(h_o);
        myScript44.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k_4];
        float[] frameMatrix = new float[h_i * w_i * c_i_4];

        int delta_c = (c_i_4 - c_i) / group;
        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript44.set_In_Blob(frameAllocation);

            myScript44.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_4; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_4 / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_4 / group) && (i < n_k_4 - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_4; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_4 / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_4 / group) && (i < n_k_4 - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript44.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float4    *****   Output: Float8
    private float[][][][] convLayerRolledParInF4OutF8(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        //check channel count
        int c_i_4 = c_i;
        if (c_i % 4 != 0)
            c_i_4 = c_i + 4 - c_i % 4;


        int n_k_8 = n_k;
        if (n_k % 8 != 0)
            n_k_8 = n_k + 8 - n_k % 8;

        int delta_n = (n_k_8 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation out1Allocation;
        Allocation out2Allocation;
        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_4 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_4(myRS), h_o * w_o * n_k_8 / 8);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        out1Allocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        out2Allocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);


        myScript48.set_Out_Alloc(out2Allocation);
        myScript48.set_c_i(c_i_4);
        myScript48.set_h_i(h_i);
        myScript48.set_w_i(w_i);
        myScript48.set_h_o(h_o);
        myScript48.set_w_o(w_o);

        // calculate the result
        float[] out1Matrix = new float[n_k_8 * h_o * w_o / 2];
        float[] out2Matrix = new float[n_k_8 * h_o * w_o / 2];
        float[] frameMatrix = new float[h_i * w_i * c_i_4];
        int delta_c = (c_i_4 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i][j][k];
                        }
            }

            frameAllocation.copyFrom(frameMatrix);
            myScript48.set_In_Blob(frameAllocation);

            myScript48.forEach_root(out1Allocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_8 / 2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (2 * i < n_k_8 / group - delta_n) {
                                outputBlob[n - 1][2 * i][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i][j][k] < 0)
                                        outputBlob[n - 1][2 * i][j][k] = 0;
                                }
                            } else if ((2 * i >= n_k_8 / group) && (2 * i < n_k_8 - delta_n)) {
                                outputBlob[n - 1][2 * i - delta_n][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][2 * i - delta_n][j][k] = 0;
                                }
                            }

                            if (2 * i + 1 < n_k_8 / group - delta_n) {
                                outputBlob[n - 1][2 * i + 1][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i + 1][j][k] < 0)
                                        outputBlob[n - 1][2 * i + 1][j][k] = 0;
                                }
                            } else if ((2 * i + 1 >= n_k_8 / group) && (2 * i + 1 < n_k_8 - delta_n)) {
                                outputBlob[n - 1][2 * i + 1 - delta_n][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i + 1 - delta_n][j][k] < 0)
                                        outputBlob[n - 1][2 * i + 1 - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            out1Allocation.copyTo(out1Matrix);
            out2Allocation.copyTo(out2Matrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_8 / 2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (2 * i < n_k_8 / group - delta_n) {
                                outputBlob[n][2 * i][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i][j][k] < 0)
                                        outputBlob[n][2 * i][j][k] = 0;
                                }
                            } else if ((2 * i >= n_k_8 / group) && (2 * i < n_k_8 - delta_n)) {
                                outputBlob[n][2 * i - delta_n][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i - delta_n][j][k] < 0)
                                        outputBlob[n][2 * i - delta_n][j][k] = 0;
                                }
                            }

                            if (2 * i + 1 < n_k_8 / group - delta_n) {
                                outputBlob[n][2 * i + 1][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i + 1][j][k] < 0)
                                        outputBlob[n][2 * i + 1][j][k] = 0;
                                }
                            } else if ((2 * i + 1 >= n_k_8 / group) && (2 * i + 1 < n_k_8 - delta_n)) {
                                outputBlob[n][2 * i + 1 - delta_n][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i + 1 - delta_n][j][k] < 0)
                                        outputBlob[n][2 * i + 1 - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        out1Allocation.destroy();
        out2Allocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript48.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float8    *****   Output: Float1
    private float[][][][] convLayerRolledParInF8OutF1(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_8 = c_i;
        if (c_i % (8 * group) != 0)
            c_i_8 = c_i + (8 * group) - c_i % (8 * group);

        int delta_n = (n_k - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_8 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32(myRS), h_o * w_o * n_k);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);


        myScript81.set_c_i(c_i_8);
        myScript81.set_h_i(h_i);
        myScript81.set_w_i(w_i);
        myScript81.set_h_o(h_o);
        myScript81.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k];
        float[] frameMatrix = new float[h_i * w_i * c_i_8];
        int delta_c = (c_i_8 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript81.set_In_Blob(frameAllocation);

            myScript81.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k + k * n_k + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k / group) && (i < n_k - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k + k * n_k + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k + k * n_k + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k / group) && (i < n_k - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k + k * n_k + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript81.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float8    *****   Output: Float2
    private float[][][][] convLayerRolledParInF8OutF2(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_8 = c_i;
        if (c_i % (8 * group) != 0)
            c_i_8 = c_i + (8 * group) - c_i % (8 * group);

        int n_k_2 = n_k;
        if (n_k % (2 * group) != 0)
            n_k_2 = n_k + (2 * group) - n_k % (2 * group);

        int delta_n = (n_k_2 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_8 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_2(myRS), h_o * w_o * n_k_2 / 2);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        myScript82.set_c_i(c_i_8);
        myScript82.set_h_i(h_i);
        myScript82.set_w_i(w_i);
        myScript82.set_h_o(h_o);
        myScript82.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k_2];
        float[] frameMatrix = new float[h_i * w_i * c_i_8];
        int delta_c = (c_i_8 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript82.set_In_Blob(frameAllocation);

            myScript82.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_2 / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_2 / group) && (i < n_k_2 - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_2 / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_2 / group) && (i < n_k_2 - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k_2 + k * n_k_2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript82.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float8    *****   Output: Float4
    private float[][][][] convLayerRolledParInF8OutF4(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_8 = c_i;
        if (c_i % (8 * group) != 0)
            c_i_8 = c_i + (8 * group) - c_i % (8 * group);

        int n_k_4 = n_k;
        if (n_k % (4 * group) != 0)
            n_k_4 = n_k + (4 * group) - n_k % (4 * group);

        int delta_n = (n_k_4 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation outAllocation;

        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_8 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_4(myRS), h_o * w_o * n_k_4 / 4);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);


        myScript84.set_c_i(c_i_8);
        myScript84.set_h_i(h_i);
        myScript84.set_w_i(w_i);
        myScript84.set_h_o(h_o);
        myScript84.set_w_o(w_o);

        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k_4];
        float[] frameMatrix = new float[h_i * w_i * c_i_8];
        int delta_c = (c_i_8 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            myScript84.set_In_Blob(frameAllocation);

            myScript84.forEach_root(outAllocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_8; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_8 / group - delta_c) && (i < c_i_8 / group)) || (i >= c_i_8 - delta_c))
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = 0;
                            else if (i >= c_i_8 / group)
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_8 + k * c_i_8 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_4; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_4 / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_4 / group) && (i < n_k_4 - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_4; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_4 / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_4 / group) && (i < n_k_4 - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k_4 + k * n_k_4 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript84.destroy();

        // return the result
        return outputBlob;
    }

    // Input: Float8    *****   Output: Float8
    private float[][][][] convLayerRolledParInF8OutF8(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_4 = c_i;
        if (c_i % (8 * group) != 0)
            c_i_4 = c_i + (8 * group) - c_i % (8 * group);

        int n_k_8 = n_k;
        if (n_k % (8 * group) != 0)
            n_k_8 = n_k + (8 * group) - n_k % (8 * group);

        int delta_n = (n_k_8 - n_k) / group;

        //initialize Renderscript
        Type inputType, outType;
        Allocation frameAllocation;
        Allocation out1Allocation;
        Allocation out2Allocation;
        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_4 * h_i * w_i / 4);
        outType = Type.createX(myRS, Element.F32_4(myRS), h_o * w_o * n_k_8 / 8);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        out1Allocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        out2Allocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);


        myScript88.set_Out_Alloc(out2Allocation);
        myScript88.set_c_i(c_i_4);
        myScript88.set_h_i(h_i);
        myScript88.set_w_i(w_i);
        myScript88.set_h_o(h_o);
        myScript88.set_w_o(w_o);

        // calculate the result
        float[] out1Matrix = new float[n_k_8 * h_o * w_o / 2];
        float[] out2Matrix = new float[n_k_8 * h_o * w_o / 2];
        float[] frameMatrix = new float[h_i * w_i * c_i_4];
        int delta_c = (c_i_4 - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n][i][j][k];
                        }
            }

            frameAllocation.copyFrom(frameMatrix);
            myScript88.set_In_Blob(frameAllocation);

            myScript88.forEach_root(out1Allocation);

            if (n < n_i - 1) {
                for (int i = 0; i < c_i_4; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_4 / group - delta_c) && (i < c_i_4 / group)) || (i >= c_i_4 - delta_c))
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = 0;
                            else if (i >= c_i_4 / group)
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_4 + k * c_i_4 + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_8 / 2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (2 * i < n_k_8 / group - delta_n) {
                                outputBlob[n - 1][2 * i][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i][j][k] < 0)
                                        outputBlob[n - 1][2 * i][j][k] = 0;
                                }
                            } else if ((2 * i >= n_k_8 / group) && (2 * i < n_k_8 - delta_n)) {
                                outputBlob[n - 1][2 * i - delta_n][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][2 * i - delta_n][j][k] = 0;
                                }
                            }

                            if (2 * i + 1 < n_k_8 / group - delta_n) {
                                outputBlob[n - 1][2 * i + 1][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i + 1][j][k] < 0)
                                        outputBlob[n - 1][2 * i + 1][j][k] = 0;
                                }
                            } else if ((2 * i + 1 >= n_k_8 / group) && (2 * i + 1 < n_k_8 - delta_n)) {
                                outputBlob[n - 1][2 * i + 1 - delta_n][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][2 * i + 1 - delta_n][j][k] < 0)
                                        outputBlob[n - 1][2 * i + 1 - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            out1Allocation.copyTo(out1Matrix);
            out2Allocation.copyTo(out2Matrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_8 / 2; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (2 * i < n_k_8 / group - delta_n) {
                                outputBlob[n][2 * i][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i][j][k] < 0)
                                        outputBlob[n][2 * i][j][k] = 0;
                                }
                            } else if ((2 * i >= n_k_8 / group) && (2 * i < n_k_8 - delta_n)) {
                                outputBlob[n][2 * i - delta_n][j][k] = out1Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i - delta_n][j][k] < 0)
                                        outputBlob[n][2 * i - delta_n][j][k] = 0;
                                }
                            }

                            if (2 * i + 1 < n_k_8 / group - delta_n) {
                                outputBlob[n][2 * i + 1][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i + 1][j][k] < 0)
                                        outputBlob[n][2 * i + 1][j][k] = 0;
                                }
                            } else if ((2 * i + 1 >= n_k_8 / group) && (2 * i + 1 < n_k_8 - delta_n)) {
                                outputBlob[n][2 * i + 1 - delta_n][j][k] = out2Matrix[j * w_o * n_k_8 / 2 + k * n_k_8 / 2 + i];
                                if (nonLinear) {
                                    if (outputBlob[n][2 * i + 1 - delta_n][j][k] < 0)
                                        outputBlob[n][2 * i + 1 - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        out1Allocation.destroy();
        out2Allocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
            myScript88.destroy();

        // return the result
        return outputBlob;
    }


    ///////////////////////////////Kernel Initialization Functions//////////////////////////////////
    private void initKernelF4F1(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_4 = c_k;
        if (c_k % 4 != 0)
            c_k_4 = c_k + 4 - c_k % 4;

        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k * c_k_4 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32(myRS), n_k);

        float[] kernelMatrix = new float[n_k * h_k * w_k * c_k_4];
        float[] biasArray = new float[n_k];
        int delta_n = (n_k - n_k) / group;
        for (int i = 0; i < n_k; i++)
            for (int j = 0; j < c_k_4; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k / group - delta_n) && (i < n_k / group)) || (i >= n_k - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = 0;
                        else if (i >= n_k / group)
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k; i++) {
            if (((i >= n_k / group - delta_n) && (i < n_k / group)) || (i >= n_k - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }

        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript41 = new ScriptC_convRolledInF4OutF1(myRS);
        myScript41.set_Bias_Blob(biasAllocation);
        myScript41.set_Kernel_Blob(kernelAllocation);
        myScript41.set_n_k(n_k);
        myScript41.set_c_k(c_k_4);
        myScript41.set_h_k(h_k);
        myScript41.set_w_k(w_k);
        myScript41.set_pad_x(pad[0]);
        myScript41.set_pad_y(pad[1]);
        myScript41.set_stride_x(stride[0]);
        myScript41.set_stride_y(stride[1]);
        myScript41.set_group(group);

    }

    private void initKernelF4F2(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_4 = c_k;
        if (c_k % 4 != 0)
            c_k_4 = c_k + 4 - c_k % 4;

        int n_k_2 = n_k;
        if (n_k % 2 != 0)
            n_k_2 = n_k + 2 - n_k % 2;


        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_2 * c_k_4 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32_2(myRS), n_k_2 / 2);

        float[] kernelMatrix = new float[n_k_2 * h_k * w_k * c_k_4];
        float[] biasArray = new float[n_k_2];
        int delta_n = (n_k_2 - n_k) / group;
        for (int i = 0; i < n_k_2; i++)
            for (int j = 0; j < c_k_4; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_2 / group - delta_n) && (i < n_k_2 / group)) || (i >= n_k_2 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = 0;
                        else if (i >= n_k_2 / group)
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_2; i++) {
            if (((i >= n_k_2 / group - delta_n) && (i < n_k_2 / group)) || (i >= n_k_2 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_2 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }
        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript42 = new ScriptC_convRolledInF4OutF2(myRS);
        myScript42.set_Bias_Blob(biasAllocation);
        myScript42.set_Kernel_Blob(kernelAllocation);
        myScript42.set_n_k(n_k_2);
        myScript42.set_c_k(c_k_4);
        myScript42.set_h_k(h_k);
        myScript42.set_w_k(w_k);
        myScript42.set_pad_x(pad[0]);
        myScript42.set_pad_y(pad[1]);
        myScript42.set_stride_x(stride[0]);
        myScript42.set_stride_y(stride[1]);
        myScript42.set_group(group);

    }

    private void initKernelF4F4(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_4 = c_k;
        if (c_k % 4 != 0)
            c_k_4 = c_k + 4 - c_k % 4;

        int n_k_4 = n_k;
        if (n_k % 4 != 0)
            n_k_4 = n_k + 4 - n_k % 4;


        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_4 * c_k_4 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32_4(myRS), n_k_4 / 4);

        float[] kernelMatrix = new float[n_k_4 * h_k * w_k * c_k_4];
        float[] biasArray = new float[n_k_4];
        int delta_n = (n_k_4 - n_k) / group;
        for (int i = 0; i < n_k_4; i++)
            for (int j = 0; j < c_k_4; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_4 / group - delta_n) && (i < n_k_4 / group)) || (i >= n_k_4 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = 0;
                        else if (i >= n_k_4 / group)
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_4; i++) {
            if (((i >= n_k_4 / group - delta_n) && (i < n_k_4 / group)) || (i >= n_k_4 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_4 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }
        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript44 = new ScriptC_convRolledInF4OutF4(myRS);
        myScript44.set_Bias_Blob(biasAllocation);
        myScript44.set_Kernel_Blob(kernelAllocation);
        myScript44.set_n_k(n_k_4);
        myScript44.set_c_k(c_k_4);
        myScript44.set_h_k(h_k);
        myScript44.set_w_k(w_k);
        myScript44.set_pad_x(pad[0]);
        myScript44.set_pad_y(pad[1]);
        myScript44.set_stride_x(stride[0]);
        myScript44.set_stride_y(stride[1]);
        myScript44.set_group(group);

    }

    private void initKernelF4F8(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_4 = c_k;
        if (c_k % 4 != 0)
            c_k_4 = c_k + 4 - c_k % 4;

        int n_k_8 = n_k;
        if (n_k % 8 != 0)
            n_k_8 = n_k + 8 - n_k % 8;


        Type kernelType, biasType;
        Allocation kernelAllocation;
        Allocation biasAllocation;
        kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_8 * c_k_4 * h_k * w_k / 4);
        biasType = Type.createX(myRS, Element.F32_4(myRS), n_k_8 / 4);

        float[] kernelMatrix = new float[n_k_8 * h_k * w_k * c_k_4];
        float[] biasArray = new float[n_k_8];

        int delta_n = (n_k_8 - n_k) / group;
        for (int i = 0; i < n_k_8; i++)
            for (int j = 0; j < c_k_4; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_8 / group - delta_n) && (i < n_k_8 / group)) || (i >= n_k_8 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = 0;
                        else if (i >= n_k_8 / group)
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_4 + k * w_k * c_k_4 + l * c_k_4 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_8; i++) {
            if (((i >= n_k_8 / group - delta_n) && (i < n_k_8 / group)) || (i >= n_k_8 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_8 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }


        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript48 = new ScriptC_convRolledInF4OutF8(myRS);
        myScript48.set_Bias_Blob(biasAllocation);
        myScript48.set_Kernel_Blob(kernelAllocation);
        myScript48.set_n_k(n_k_8);
        myScript48.set_c_k(c_k_4);
        myScript48.set_h_k(h_k);
        myScript48.set_w_k(w_k);
        myScript48.set_pad_x(pad[0]);
        myScript48.set_pad_y(pad[1]);
        myScript48.set_stride_x(stride[0]);
        myScript48.set_stride_y(stride[1]);
        myScript48.set_group(group);

    }

    private void initKernelF8F1(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_8 = c_k;
        if (c_k % 8 != 0)
            c_k_8 = c_k + 8 - c_k % 8;

        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k * c_k_8 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32(myRS), n_k);

        float[] kernelMatrix = new float[n_k * h_k * w_k * c_k_8];
        float[] biasArray = new float[n_k];
        int delta_n = (n_k - n_k) / group;
        for (int i = 0; i < n_k; i++)
            for (int j = 0; j < c_k_8; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k / group - delta_n) && (i < n_k / group)) || (i >= n_k - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = 0;
                        else if (i >= n_k / group)
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k; i++) {
            if (((i >= n_k / group - delta_n) && (i < n_k / group)) || (i >= n_k - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }

        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript81 = new ScriptC_convRolledInF8OutF1(myRS);
        myScript81.set_Bias_Blob(biasAllocation);
        myScript81.set_Kernel_Blob(kernelAllocation);
        myScript81.set_n_k(n_k);
        myScript81.set_c_k(c_k_8);
        myScript81.set_h_k(h_k);
        myScript81.set_w_k(w_k);
        myScript81.set_pad_x(pad[0]);
        myScript81.set_pad_y(pad[1]);
        myScript81.set_stride_x(stride[0]);
        myScript81.set_stride_y(stride[1]);
        myScript81.set_group(group);
    }

    private void initKernelF8F2(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_8 = c_k;
        if (c_k % 8 != 0)
            c_k_8 = c_k + 8 - c_k % 8;

        int n_k_2 = n_k;
        if (n_k % 2 != 0)
            n_k_2 = n_k + 2 - n_k % 2;


        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_2 * c_k_8 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32_2(myRS), n_k_2 / 2);

        float[] kernelMatrix = new float[n_k_2 * h_k * w_k * c_k_8];
        float[] biasArray = new float[n_k_2];
        int delta_n = (n_k_2 - n_k) / group;
        for (int i = 0; i < n_k_2; i++)
            for (int j = 0; j < c_k_8; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_2 / group - delta_n) && (i < n_k_2 / group)) || (i >= n_k_2 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = 0;
                        else if (i >= n_k_2 / group)
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_2; i++) {
            if (((i >= n_k_2 / group - delta_n) && (i < n_k_2 / group)) || (i >= n_k_2 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_2 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }
        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript82 = new ScriptC_convRolledInF8OutF2(myRS);
        myScript82.set_Bias_Blob(biasAllocation);
        myScript82.set_Kernel_Blob(kernelAllocation);
        myScript82.set_n_k(n_k_2);
        myScript82.set_c_k(c_k_8);
        myScript82.set_h_k(h_k);
        myScript82.set_w_k(w_k);
        myScript82.set_pad_x(pad[0]);
        myScript82.set_pad_y(pad[1]);
        myScript82.set_stride_x(stride[0]);
        myScript82.set_stride_y(stride[1]);
        myScript82.set_group(group);
    }

    private void initKernelF8F4(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_8 = c_k;
        if (c_k % 8 != 0)
            c_k_8 = c_k + 8 - c_k % 8;

        int n_k_4 = n_k;
        if (n_k % 4 != 0)
            n_k_4 = n_k + 4 - n_k % 4;


        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_4 * c_k_8 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32_4(myRS), n_k_4 / 4);

        float[] kernelMatrix = new float[n_k_4 * h_k * w_k * c_k_8];
        float[] biasArray = new float[n_k_4];
        int delta_n = (n_k_4 - n_k) / group;
        for (int i = 0; i < n_k_4; i++)
            for (int j = 0; j < c_k_8; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_4 / group - delta_n) && (i < n_k_4 / group)) || (i >= n_k_4 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = 0;
                        else if (i >= n_k_4 / group)
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_4; i++) {
            if (((i >= n_k_4 / group - delta_n) && (i < n_k_4 / group)) || (i >= n_k_4 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_4 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }
        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript84 = new ScriptC_convRolledInF8OutF4(myRS);
        myScript84.set_Bias_Blob(biasAllocation);
        myScript84.set_Kernel_Blob(kernelAllocation);
        myScript84.set_n_k(n_k_4);
        myScript84.set_c_k(c_k_8);
        myScript84.set_h_k(h_k);
        myScript84.set_w_k(w_k);
        myScript84.set_pad_x(pad[0]);
        myScript84.set_pad_y(pad[1]);
        myScript84.set_stride_x(stride[0]);
        myScript84.set_stride_y(stride[1]);
        myScript84.set_group(group);

    }

    private void initKernelF8F8(float[][][][] myWeight, float[] myBias) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_8 = c_k;
        if (c_k % 8 != 0)
            c_k_8 = c_k + 8 - c_k % 8;

        int n_k_8 = n_k;
        if (n_k % 8 != 0)
            n_k_8 = n_k + 8 - n_k % 8;

        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_8 * c_k_8 * h_k * w_k / 4);
        Type biasType = Type.createX(myRS, Element.F32_4(myRS), n_k_8 / 4);


        float[] kernelMatrix = new float[n_k_8 * h_k * w_k * c_k_8];
        float[] biasArray = new float[n_k_8];

        int delta_n = (n_k_8 - n_k) / group;
        for (int i = 0; i < n_k_8; i++)
            for (int j = 0; j < c_k_8; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_8 / group - delta_n) && (i < n_k_8 / group)) || (i >= n_k_8 - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = 0;
                        else if (i >= n_k_8 / group)
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_8 + k * w_k * c_k_8 + l * c_k_8 + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_8; i++) {
            if (((i >= n_k_8 / group - delta_n) && (i < n_k_8 / group)) || (i >= n_k_8 - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_8 / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }


        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);

        myScript88 = new ScriptC_convRolledInF8OutF8(myRS);
        myScript88.set_Bias_Blob(biasAllocation);
        myScript88.set_Kernel_Blob(kernelAllocation);
        myScript88.set_n_k(n_k_8);
        myScript88.set_c_k(c_k_8);
        myScript88.set_h_k(h_k);
        myScript88.set_w_k(w_k);
        myScript88.set_pad_x(pad[0]);
        myScript88.set_pad_y(pad[1]);
        myScript88.set_stride_x(stride[0]);
        myScript88.set_stride_y(stride[1]);
        myScript88.set_group(group);
    }

    /////////////////////////////////////////Tuning Function////////////////////////////////////////
    private Object tuneFunction(float[][][][] input) {
        Log.d("CNNdroid", "layers." + name + ": Tuning process is starting...");
        long tuneTime = System.currentTimeMillis();

        Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
        float[][][][] myWeight = (float[][][][]) objects[0];
        float[] myBias = (float[]) objects[1];
        tune = false;
        long[] time = new long[]{0, 0, 0, 0};
        long temp;
        int c_i = input[0].length;
        float[][][][] tuneInput = new float[1][c_i][input[0][0].length][input[0][0][0].length];
        tuneInput[0] = input[0];

        if (c_i < 5) {
            for (int i = 0; i < 4; i++) {
                temp = System.currentTimeMillis();
                initKernelF4F1(myWeight, myBias);
                convLayerRolledParInF4OutF1(tuneInput, myWeight, true);
                time[0] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF4F2(myWeight, myBias);
                convLayerRolledParInF4OutF2(tuneInput, myWeight, true);
                time[1] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF4F4(myWeight, myBias);
                convLayerRolledParInF4OutF4(tuneInput, myWeight, true);
                time[2] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF4F8(myWeight, myBias);
                convLayerRolledParInF4OutF8(tuneInput, myWeight, true);
                time[3] += System.currentTimeMillis() - temp;
            }

            int min = 0;
            for (int i = 0; i < 4; i++)
                if (time[i] <= time[min])
                    min = i;

            algorithm = names[min];
        } else {
            for (int i = 0; i < 3; i++) {
                temp = System.currentTimeMillis();
                initKernelF8F1(myWeight, myBias);
                convLayerRolledParInF8OutF1(tuneInput, myWeight, true);
                time[0] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF8F2(myWeight, myBias);
                convLayerRolledParInF8OutF2(tuneInput, myWeight, true);
                time[1] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF8F4(myWeight, myBias);
                convLayerRolledParInF8OutF4(tuneInput, myWeight, true);
                time[2] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelF8F8(myWeight, myBias);
                convLayerRolledParInF8OutF8(tuneInput, myWeight, true);
                time[3] += System.currentTimeMillis() - temp;
            }

            int min = 0;
            for (int i = 0; i < 4; i++)
                if (time[i] <= time[min])
                    min = i;

            algorithm = names[min + 4];
        }

        initKernelF4F8(myWeight, myBias);
        Object output = convLayerRolledParInF4OutF8(input, myWeight, true);

        writeFile(algorithm);
        if(loadParamsAtStart) {
            weight = myWeight;
            bias = myBias;
            switch (algorithm) {
                case "F4F1":
                    initKernelF4F1(weight, bias);
                    break;
                case "F4F2":
                    initKernelF4F2(weight, bias);
                    break;
                case "F4F4":
                    initKernelF4F4(weight, bias);
                    break;
                case "F4F8":
                    initKernelF4F8(weight, bias);
                    break;
                case "F8F1":
                    initKernelF8F1(weight, bias);
                    break;
                case "F8F2":
                    initKernelF8F2(weight, bias);
                    break;
                case "F8F4":
                    initKernelF8F4(weight, bias);
                    break;
                case "F8F8":
                    initKernelF8F8(weight, bias);
                    break;
            }
        }
        tuneTime = System.currentTimeMillis() - tuneTime;
        Log.d("CNNdroid", "layers." + name + ": Tuning process finished in " + tuneTime + "mS!");
        return output;
    }

    ////////////////////////////////////////Local Functions/////////////////////////////////////////
    private Object invokeFunctions(Object input, float[][][][] myWeight, float[] myBias, boolean destroy)
    {
        Object output = null;
        long runTime = System.currentTimeMillis();

        if (!parallel)
            output = convLayerRolledSeq((float[][][][]) input, myWeight, myBias, pad, stride, group);
        else {
            if (tune) {
                output = tuneFunction((float[][][][]) input);
            }
            else {
                switch (algorithm) {
                    case "F4F1":
                        output = convLayerRolledParInF4OutF1((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F4F2":
                        output = convLayerRolledParInF4OutF2((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F4F4":
                        output = convLayerRolledParInF4OutF4((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F4F8":
                        output = convLayerRolledParInF4OutF8((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F8F1":
                        output = convLayerRolledParInF8OutF1((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F8F2":
                        output = convLayerRolledParInF8OutF2((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F8F4":
                        output = convLayerRolledParInF8OutF4((float[][][][]) input, myWeight, destroy);
                        break;
                    case "F8F8":
                        output = convLayerRolledParInF8OutF8((float[][][][]) input, myWeight, destroy);
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

