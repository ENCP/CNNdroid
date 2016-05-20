# .caffemodel file path:
MODEL_FILE = '/home/user/Desktop/cifar10/cifar10_quick_iter_5000.caffemodel'
# .prototxt file path:
MODEL_NET = '/home/user/Desktop/cifar10/cifar10_quick.prototxt'
# Saving path:
SAVE_TO = '/home/user/Desktop/cifar10/'

# Set True if you want to get parameters:
GET_PARAMS = True
# Set True if you want to get blobs:
GET_BLOBS = False

import sys
import os
import numpy as np
import msgpack

# Make sure that Caffe is on the python path:
# Caffe installation path:
caffe_root = '/home/user/caffe/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append(os.path.abspath('.'))
import caffe


def get_blobs(net):
    net_blobs = {'info': 'net.blobs data'}
    for key in net.blobs.iterkeys():
        print('Getting blob: ' + key)
        net_blobs[key] = net.blobs[key].data

    return net_blobs

def get_params(net):
    net_params = {'info': 'net.params data'}
    for key in net.params.iterkeys():
        print('Getting parameters: ' + key)
        if type(net.params[key]) is not caffe._caffe.BlobVec:
            net_params[key] = net.params[key].data
        else:
            net_params[key] = [net.params[key][0].data, net.params[key][1].data]

    return net_params


# Open a model:
caffe.set_mode_cpu()
net = caffe.Net(MODEL_NET, MODEL_FILE, caffe.TEST)
net.forward()

# Extract the model:
if GET_BLOBS:
    blobs = get_blobs(net)
if GET_PARAMS:
    params = get_params(net)

save_list = []

# Write blobs:
if GET_BLOBS:
    save_list.append('***BLOBS***')
    for b in blobs.iterkeys():
        print('Saving blob: ' + b)
        if type(blobs[b]) is np.ndarray:
            save_list.append(b)
            buf = msgpack.packb(blobs[b].tolist(), use_single_float = True)
            with open(SAVE_TO + 'model_blob_'+ b + '.msg', 'wb') as f:
                f.write(buf)
        else:
            print('Blob ' + b + ' not saved.')

# Write parameters:
if GET_PARAMS:
    save_list.append('***PARAMS***')
    for b in params.iterkeys():
        print('Saving parameters: ' + b)
        if type(params[b]) is np.ndarray:
            save_list.append(b)
            buf = msgpack.packb(params[b].tolist(), use_single_float = True)
            with open(SAVE_TO + 'model_param_'+ b + '.msg', 'wb') as f:
                f.write(buf)
        elif type(params[b]) is list:
            save_list.append(b)
            if len(params[b][0].shape) == 4:			# for the convolution layers
               buf1 = msgpack.packb(params[b][0].tolist(), use_single_float = True)
            elif len(params[b][0].shape) == 2:			# for the fully-connected layers
               buf1 = msgpack.packb(params[b][0].ravel().tolist(), use_single_float = True)
            buf2 = msgpack.packb(params[b][1].tolist(), use_single_float = True)
            with open(SAVE_TO + 'model_param_'+ b + '.msg', 'wb') as f:
                f.write(buf1)
                f.write(buf2)
        else:
            print('Parameters ' + b + ' not saved.')

print('Saved data:')
print(save_list)
