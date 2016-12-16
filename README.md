# CNNdroid
CNNdroid is an open source library for execution of trained convolutional neural networks on Android devices.
The main highlights of CNNdroid are as follows:
* Support for nearly all CNN layer types.
* Compatible with CNN models trained by common desktop/server libraries, namely, [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/) and [Theano](https://github.com/Theano/Theano) (Developers can easily convert the trained models to CNNdroid format using the provided scripts).
* Easy to configure and integrate into any Android app in Android SDK without additional software requirements.
* User-specified maximum memory usage.
* GPU or CPU acceleration of supported CNN layers.
* Automatic tuning of performance.
* Up to 60X speedup and up to 130X energy saving on current mobile devices.

For more information about the library and installation guide, please refer to the [user guide](CNNdroid Complete Developers Guide and Installation Instruction.pdf).

Please cite CNNdroid in your publications if it helps your research:
```
@inproceedings{cnndroid2016,
 author = {Latifi Oskouei, Seyyed Salar and Golestani, Hossein and Hashemi, Matin and Ghiasi, Soheil},
 title = {CNNdroid: GPU-Accelerated Execution of Trained Deep Convolutional Neural Networks on Android},
 booktitle = {Proceedings of the 2016 ACM on Multimedia Conference},
 series = {MM '16},
 year = {2016},
 location = {Amsterdam, The Netherlands},
 pages = {1201--1205}
}
```
