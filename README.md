# benchmark-FP32-FP16-INT8-in-pytorch
Benchmark inference speed of CNNs with various quantization methods in Pytorch!

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
