# CNNdroid
CNNdroid is an open source library for execution of trained convolutional neural networks on Android devices.
The main highlights of CNNdroid are as follows:
* Support for all major CNN layer types.
* Compatible with CNN models trained by common desktop libraries, namely, [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/) and [Theano](https://github.com/Theano/Theano) (Developers can easily port the trained models to Android by the help of provided scripts).
* Easy to configure and integrate into any Android app without additional software requirements.
* User-specified maximum memory usage.
* Acceleration on both GPU and CPU.
* Automatic tuning of performance.
* Up to 60X speedup and up to 130X energy saving on current mobile devices

For more information about the library and installation guide, please refer to the [user guide](CNNdroid Complete Developers Guide and Installation Instruction.pdf).

Please cite CNNdroid in your publications if it helps your research:
@article{cnndroid2016,
  Author = {Seyyed Salar Latifi Oskouei and Hossein Golestani and Matin Hashemi and Soheil Ghiasi},
  Journal = {ACM Multimedia},
  Title = {CNNdroid: GPU-Accelerated Execution of Trained Deep Convolutional Neural Networks on Android},
  Year = {2016}
}
