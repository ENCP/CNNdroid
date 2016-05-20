#List of functions:
#   1- save_numpy_parameters
#   2- save_tensor_parameters


import numpy as np
import msgpack as mp


# Saving path:
SAVE_TO = '/home/user/Desktop/'


#Function to save parameters num numpy array format
#if weight_parameter is weight of a convolution layer:
#   Dimensions: (next layer features, previous layer features, kernel height, kernel width)
def save_numpy_parameters(weight_parameter, bias_parameter, layer_name):
    weight = weight_parameter.astype('float32')
    bias = bias_parameter.astype('float32')
    if len(weight.shape) is 4:
        buf1 = mp.packb(weight.tolist(), use_single_float = True)
    else:
        buf1 = mp.packb(weight.flatten().tolist(), use_single_float = True)
    buf2 = mp.packb(bias.tolist(), use_single_float = True)
    with open(SAVE_TO + 'model_param_'+ layer_name + '.msg', 'wb') as f:
        f.write(buf1)
        f.write(buf2)    
    print('Saved data:')
    print(layer_name + '.weight & ' + layer_name + ".bias in 'model_param_" + layer_name + ".msg'")



def save_tensor_parameters(weight_parameter, bias_parameter, layer_name):
    np_weight = np.asanyarray(weight_parameterb.eval())
    np_bias = np.asanyarray(bias_parameterb.eval())
    save_numpy_parameters(np_weight, np_bias, layer_name)

