# Benchmark-FP32-FP16-INT8-with-TensorRT
Benchmark inference speed of CNNs with various quantization methods with TensorRT!

:star: if it helps you.

# Image classification

Run:
`inference_tensorrt.py`

## Hardware:Jetson Nano.
Models are converted to TensorRT unless noted.

Latency of image inference (1,3,256,256) [ms]

|          | FP32 | FP16 | INT8 |
|:--------:|------|:----:|------|
| resnet18 | 26   |  18  |      |
| resnet34 | 48   |  30  |      |
| resnet50 | 79   | 42   |      |

Jetson Nano does not support INT8..

## Hardware:Jetson Xavier.

Models are converted to TensorRT unless noted.

Latency of image inference (1,3,256,256) [ms]

|      | resnet18 | resnet34 | resnet50 |
|------|----------|----------|----------|
| Raw  | 11       | 12       | 16       |
| FP32 | 3.8      | 5.6      | 9.9      |
| FP16 | 2.1      | 3.3      | 4.4      |
| INT8 | 1.7      | 2.7      | 3.0     |

# Image segmentation
![beatles](imgs/addtensorrt_FP32.jpg)
## Hardware:Jetson Xavier.

Models are converted to TensorRT unless noted.

Latency of image inference (1,3,512,512) [ms]

|      | fcn_resnet50 | fcn_resnet101 | deeplabv3_resnet50 | deeplabv3_resnet101 |
|------|--------------|---------------|--------------------|---------------------|
| Raw  | 200          | 344           | 281                | 426                 |
| FP32 | 173          | 290           | 252                | 366                 |
| FP16 | 36           | 57            | 130                | 151                 |
| INT8 | 21           | 32            | 97                 | 108                 |

## Hardware:Jetson Nano.

Latency of image inference (1,3,256,256) [ms]

|      | fcn_resnet50 | 
|------|--------------|
| Raw  | 6800          | 
| FP32 | 767          | 
| FP16 | 40           | 
| INT8 | NA           | 

# Hardware setup
The hardware setup seems tricky.

* Install pytorch

https://forums.developer.nvidia.com/t/pytorch-for-jetson-nano-version-1-4-0-now-available/72048

**The stable version for Jetson nano seems to be torch==1.1**

**For Xavier, torch==1.3 worked fine for me.**

* Install torchvision

I followed this instruction and installed torchvision==0.3.0

https://medium.com/hackers-terminal/installing-pytorch-torchvision-on-nvidias-jetson-tx2-81591d03ce32

```bash
sudo apt-get install libjpeg-dev zlib1g-dev
git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
```

* Install torch2trt

Followed readme.

https://github.com/NVIDIA-AI-IOT/torch2trt

```bash
sudo apt-get install libprotobuf* protobuf-compiler ninja-build
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins 
```
