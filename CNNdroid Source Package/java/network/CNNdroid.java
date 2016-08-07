package network;

import android.renderscript.RenderScript;
import android.util.Log;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import layers.Accuracy;
import layers.Convolution;
import layers.FullyConnected;
import layers.LayerInterface;
import layers.LocalResponseNormalization;
import layers.NonLinear;
import layers.Pooling;
import layers.Softmax;

public class CNNdroid {

    private static final long MAX_PARAM_SIZE = 419430400;
    private static final String tuningFolder = "CNNdroid_Tuning";

    private ArrayList<LayerInterface> layers;   // list of the network layers
    private boolean parallel;                   // implementation method (parallel or sequential)
    private boolean autoTuning;                 // auto-tuning (on or off)
    private long allocatedRAM = -1;             // size of RAM allocated to the parameters
    private boolean[] loadtAtStart;             // whether or not the parameters should be loaded at start-up
    private int layerCounter = 0;               // counter for the layers which have parameters
    private String rootDir;                     // the root directory of network parameters file
    private String netStructureFile;            // the directory of the network definition file
    private RenderScript myRS;                  // RenderScript object
    private LayerInterface lastLayer = null;    // the last constructed layer
    private boolean[] necessaryDefinition;      // execution_mode, auto_tuning

    public CNNdroid(RenderScript myRS, String netStructureFile) throws Exception {
        this.myRS = myRS;
        this.netStructureFile = netStructureFile;

        necessaryDefinition = new boolean[2];

        layers = new ArrayList<>();
        preParse();
        parse();
        File f = new File(rootDir + tuningFolder);
        if (!f.exists())
            f.mkdir();
    }

    // support for 3d input
    public Object compute(Object input) {
        Object output;
        if (input.getClass().toString().equals("class [[[[F"))
            output = input;
        else if (input.getClass().toString().equals("class [[[F")) {
            float[][][][] newInput = new float[1][][][];
            newInput[0] = (float[][][]) input;
            output = newInput;
        } else {
            Log.d("CNNdroid", "Error: input type is not supported");
            return null;
        }


        for (int i = 0 ; i < layers.size() ; i++) {
            Object temp = output;
            output = layers.get(i).compute(output);
        }

        return output;
    }

    // Determine whether or not the parameters should be loaded at start-up.
    private void preParse() throws Exception {
        File f = new File(netStructureFile);
        List<Long> paramSize = new ArrayList<>();
        Scanner s;
        String root = "";

        s = new Scanner(f);
        while (s.hasNextLine()) {
            String str = s.nextLine();
            str = str.trim();
            String strLow = str.toLowerCase();
            if (strLow.startsWith("root_directory")) {
                str = str.substring(14);
                root = deriveStr(str);
            }
            else if (strLow.startsWith("allocated_ram")) {
                str = str.substring(13);
                long l = Long.parseLong(deriveNum(str));
                if (l < MAX_PARAM_SIZE)
                    allocatedRAM = l * 1024 * 1024;
                else
                    allocatedRAM = MAX_PARAM_SIZE;
            }
            else if (strLow.startsWith("parameters_file")) {
                str = str.substring(15);
                String fName = deriveStr(str);
                File pf = new File(root + fName);
                if (pf.exists())
                    paramSize.add(pf.length());
                else {
                    Log.d("CNNdroid", "Error: Missing parameters file \"" + str + "\"");
                    throw new Exception("CNNdroid parameter file does not exist.");
                }
            }
        }

        if (root.equals("")) {
            Log.d("CNNdroid", "Error: root_directory is not specified in the network structure definition file");
            throw new Exception("CNNdroid root directory is not specified.");
        }
        if (allocatedRAM == -1) {
            Log.d("CNNdroid", "Error: allocated_ram is not specified in the network structure definition file");
            throw new Exception("CNNdroid allocated RAM is not specified.");
        }

        long[] params = longArray(paramSize);
        int[] index = mergeSort(params, 0, params.length - 1);

        loadtAtStart = new boolean[params.length];

        long sum = allocatedRAM;
        for (int i = 0; i < params.length; ++i) {
            if (sum - params[index[i]] >= 0) {
                sum -= params[index[i]];
                loadtAtStart[index[i]] = true;
            }
        }
    }

    // Parse the network definition file and construct layers.
    private void parse() throws Exception {
        int layerNum = 0;
        File f = new File(netStructureFile);
        Scanner s = new Scanner(f);

        while (s.hasNextLine()) {
            String str = s.nextLine();
            str = str.trim();
            String strLow = str.toLowerCase();

            if (strLow.startsWith("root_directory")) {
                str = str.substring(14);
                str = deriveStr(str);
                if (str.equals("")) {
                    Log.d("CNNdroid", "Error: root_directory is not specified correctly in the network structure definition file");
                    throw new Exception("CNNdroid root directory is not specified correctly.");
                }
                rootDir = str;
            }
            else if (strLow.startsWith("allocated_ram")) {
                str = str.substring(13);
                str = deriveNum(str);
                if (str.equals("")) {
                    Log.d("CNNdroid", "Error: allocated_ram is not specified correctly in the network structure definition file");
                    throw new Exception("CNNdroid allocated RAM is not specified correctly.");
                }
            }
            else if (strLow.startsWith("execution_mode")) {
                strLow = strLow.substring(14);
                strLow = deriveStr(strLow);
                if (strLow.equals("parallel"))
                    parallel = true;
                else if (strLow.equals("sequential"))
                    parallel = false;
                else {
                    Log.d("CNNdroid", "Error: execution_mode is not specified correctly in the network structure definition file");
                    throw new Exception("CNNdroid execution mode is not specified correctly.");
                }
                necessaryDefinition[0] = true;
            }
            else if (strLow.startsWith("auto_tuning")) {
                strLow = strLow.substring(11);
                strLow = deriveStr(strLow);
                if (strLow.equals("on"))
                    autoTuning = true;
                else if (strLow.equals("off"))
                    autoTuning = false;
                else {
                    Log.d("CNNdroid", "Error: auto_tuning is not specified correctly in the network structure definition file");
                    throw new Exception("CNNdroid auto-tuning is not specified correctly.");
                }
                necessaryDefinition[1] = true;
            }
            else if (strLow.startsWith("layer")) {
                ++layerNum;
                str = str.substring(5);
                String tempStr;
                while (s.hasNextLine()) {
                    str += "\n";
                    tempStr = s.nextLine();
                    str += tempStr;
                    if (tempStr.contains("}"))
                        break;
                }
                if (!deriveLayer(str)) {
                    Log.d("CNNdroid", "Error: Layer number " + layerNum + " is not defined correctly in the network structure definition file");
                    throw new Exception("CNNdroid layer number " + layerNum + " is not defined correctly.");
                }
            }
            else if (strLow.equals(""))
                continue;
            else
            {
                Log.d("CNNdroid", "Error in the network structure definition file: " + str);
                throw new Exception("Error in CNNdroid network structure definition file: " + str);
            }
        }

        Log.d("#####", "Hello!");

        if (!necessaryDefinition[0]) {
            Log.d("CNNdroid", "Error: execution_mode is not specified in the network structure definition file");
            throw new Exception("CNNdroid execution mode is not specified.");
        }
        if (!necessaryDefinition[1]) {
            Log.d("CNNdroid", "Error: auto_tuning is not specified in the network structure definition file");
            throw new Exception("CNNdroid auto-tuning is not specified.");
        }
    }

    private String deriveStr(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        if (!str.startsWith("\""))
            return "";
        str = str.substring(1);

        int i = str.indexOf("\"");
        if (i == -1)
            return "";

        String retStr = str.substring(0, i);

        if (i != str.length() - 1)
        {
            str = str.substring(i);
            str = str.trim();
            if (!str.equals(""))
                return "";
        }

        return retStr;
    }

    private String deriveNum(String str) {
        str = str.trim();
        if (!str.startsWith(":"))
            return "";
        str = str.substring(1);

        str = str.trim();
        int i = str.indexOf(" ");

        if (i == -1)
            return str;

        if (!str.substring(i).trim().equals(""))
            return "";
        else
            return str.substring(0, i);
    }

    private boolean deriveLayer(String str) {
        String[] strArr = str.split("\n");

        strArr[0] = strArr[0].trim();
        strArr[1] = strArr[1].trim();

        if (strArr[0].startsWith("{"))
            strArr[0] = strArr[0].substring(1);
        else if (strArr[1].startsWith("{"))
            strArr[1] = strArr[1].substring(1);
        else
            return false;

        String endStr = strArr[strArr.length - 1];
        endStr = endStr.trim();
        if (!endStr.startsWith("}"))
            return false;
        endStr = endStr.substring(1);
        endStr = endStr.trim();
        if (!endStr.equals(""))
            return false;

        String type = "";
        String name = "";
        List<String> args = new ArrayList<String>();
        List<String> values= new ArrayList<String>();

        for (int i = 0; i < strArr.length - 1; ++i)
        {
            strArr[i] = strArr[i].trim();
            if (strArr[i].equals(""))
                continue;

            int i1 = strArr[i].indexOf(":");
            int i2 = strArr[i].indexOf(" ");
            int j = (i1 < i2) ? i1 : i2;
            String tempArg = strArr[i].substring(0, j);
            String tempValue;
            String temp = strArr[i].substring(j);
            if (temp.contains("\""))
                tempValue = deriveStr(temp);
            else
                tempValue = deriveNum(temp);

            if (tempArg.equalsIgnoreCase("name"))
                name = tempValue;
            else if (tempArg.equalsIgnoreCase("type"))
                type = tempValue;
            else
            {
                args.add(tempArg);
                values.add(tempValue);
            }
        }

        if (type.equalsIgnoreCase("Convolution"))
        {
            String parametersFile = null;
            int pad = -1;
            int stride = -1;
            int group = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("parameters_file"))
                    parametersFile = tempValue;
                else if (tempArg.equalsIgnoreCase("pad"))
                    pad = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase("stride"))
                    stride = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase("group"))
                    group = Integer.parseInt(tempValue);
                else
                    return false;
            }
            if (parametersFile == null || pad == -1 || stride == -1 || group == -1)
                return false;
            Convolution c = new Convolution(new int[]{stride, stride}, new int[]{pad, pad}, group,
                    rootDir + parametersFile, parallel, loadtAtStart[layerCounter], autoTuning, myRS, name, rootDir + tuningFolder);
            ++layerCounter;
            lastLayer = c;
            layers.add(c);
            return true;
        }
        else if (type.equalsIgnoreCase("Pooling")) {
            String pool = null;
            int kernelSize = -1;
            int pad = -1;
            int stride = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("pool"))
                    pool = tempValue;
                else if (tempArg.equalsIgnoreCase("kernel_size"))
                    kernelSize = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase("pad"))
                    pad = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase("stride"))
                    stride = Integer.parseInt(tempValue);
                else
                    return false;
            }
            if (pool == null || pad == -1 || stride == -1 || kernelSize == -1)
                return false;
            Pooling p = new Pooling(new int[]{kernelSize, kernelSize}, pool, new int[]{pad, pad},
                    new int[]{stride, stride}, parallel, autoTuning, name, rootDir + tuningFolder);
            lastLayer = p;
            layers.add(p);
            return true;
        }
        else if (type.equalsIgnoreCase("LRN")) {
            String normRegion = null;
            int localSize = -1;
            double alpha = -1.0;
            double beta = -1.0;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("norm_region"))
                    normRegion = tempValue;
                else if (tempArg.equalsIgnoreCase("local_size"))
                    localSize = Integer.parseInt(tempValue);
                else if (tempArg.equalsIgnoreCase("alpha"))
                    alpha = Double.parseDouble(tempValue);
                else if (tempArg.equalsIgnoreCase("beta"))
                    beta = Double.parseDouble(tempValue);
                else
                    return false;
            }
            if (normRegion == null || localSize == -1 || alpha == -1.0 || beta == -1.0)
                return false;
            LocalResponseNormalization lrn =  new LocalResponseNormalization(localSize, alpha, beta,
                    normRegion, parallel, autoTuning, name, rootDir + tuningFolder);
            lastLayer = lrn;
            layers.add(lrn);
            return true;
        }
        else if (type.equalsIgnoreCase("FullyConnected")) {
            String parametersFile = null;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("parameters_file"))
                    parametersFile = tempValue;
                else
                    return false;
            }
            if (parametersFile == null)
                return false;
            FullyConnected fc = new FullyConnected(rootDir + parametersFile, parallel, loadtAtStart[layerCounter], autoTuning, myRS, name, rootDir + tuningFolder);
            ++layerCounter;
            lastLayer = fc;
            layers.add(fc);
            return true;
        }
        else if (type.equalsIgnoreCase("Accuracy")) {
            String parametersFile = null;
            int topk = -1;
            for (int i = 0; i < args.size(); ++i) {
                String tempArg = args.get(i);
                String tempValue = values.get(i);
                if (tempArg.equalsIgnoreCase("parameters_file"))
                    parametersFile = tempValue;
                else if (tempArg.equalsIgnoreCase("topk"))
                    topk = Integer.parseInt(tempValue);
                else
                    return false;
            }
            if (parametersFile == null || topk == -1)
                return false;
            Accuracy a = new Accuracy(topk, rootDir + parametersFile, name);
            lastLayer = a;
            layers.add(a);
            return true;
        }
        else if (type.equalsIgnoreCase("Softmax")) {
            if (args.size() != 0)
                return false;
            Softmax sm = new Softmax(name);
            lastLayer = sm;
            layers.add(sm);
            return true;
        }
        else if (type.equalsIgnoreCase("ReLU")) {
            if (parallel) {
                if (lastLayer instanceof Convolution) {
                    ((Convolution) lastLayer).setNonLinearType(Convolution.NonLinearType.RectifiedLinearUnit);
                    return true;
                }
                else if (lastLayer instanceof FullyConnected) {
                    ((FullyConnected) lastLayer).setNonLinearType(FullyConnected.NonLinearType.RectifiedLinearUnit);
                    return true;
                }
            }
            NonLinear nl = new NonLinear(name, NonLinear.NonLinearType.RectifiedLinearUnit);
            lastLayer = nl;
            layers.add(nl);
            return true;
        }
        else
            return false;
    }

    long[] longArray(List<Long> ll)
    {
        long[] l = new long[ll.size()];
        Iterator<Long> i = ll.iterator();

        int j = 0;
        while (i.hasNext()) {
            l[j] = i.next();
            ++j;
        }

        return l;
    }

    // decrementing order
    private int[] mergeSort(long[] a, int s, int e)
    {
        if (s > e)
            return null;

        if (s == e)
            return new int[]{s};

        int m = (s + e) / 2;
        int[] index1 = mergeSort(a, s, m);
        int[] index2 = mergeSort(a, m + 1, e);

        return merge(a, index1, index2);
    }

    private int[] merge(long a[], int[] index1, int[] index2)
    {
        int l1 = index1.length;
        int l2 = index2.length;

        int[] index = new int[l1 + l2];
        int i = 0, j = 0, k = 0;

        while (k < l1 + l2)
        {
            if ((i < l1) && ((j >= l2) || (a[index1[i]] > a[index2[j]])))
            {
                index[k] = index1[i];
                ++i;
            }
            else
            {
                index[k] = index2[j];
                ++j;
            }
            ++k;
        }

        return index;
    }

}


