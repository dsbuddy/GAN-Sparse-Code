# GAN Sparse Code

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

CS583 Final Project  

GANs need very little data to learn trends and make predictions. As a result, if we could use GANs to learn the sparse codes that can then be used for later downstream tasks like: Image Classification, Object Recognition, Optical Flow Tracking, etc. The sparse codes are a small and efficient representation of the picture and would require less computational power on some of these tasks as opposed to the larger and deeper DNNs used nowadays. Additionally, the GANs could possibly outperform existing sparse code algorithms by using fewer data points.

This repository contains:

1. [GAN](Model.ipynb) of a Generative Adversarial Network that learns sparse codes for CIFAR-10 Images.
2. A file of the [dataset](test_batch) you can use to view images of CIFAR-10.
3. A file of the [dictionary](dictionary) for which the sparse codes are learned.
4. [Sparse Code Script](LCAWithColor.py) to generate the sparsse codes that match the [dictionary](dictionary).


## Install

This project uses [pip](https://pypi.org/). Go check them out if you don't have them locally installed.

```sh
$ pip3 install numpy  
$ pip3 install matplotlib  
$ pip3 install sklearn  
$ pip3 install passlib  
$ pip3 install scipy  
$ pip3 install tensorflow  
$ pip3 install keras  
$ pip3 install pillow  
```

## Usage

To run either execute LCAWithColor.py to generate sparse codes for ground truth and then run jupyter lab to run all cells of Model.ipynb:  

```sh
$ python3 LCAWithColor.py  
$ jupyter lab  
```

### Contributors

This project exists thanks to all the people who contribute. 
Daniel Schwartz and Jeff Winchell  

## License

[MIT](LICENSE)
