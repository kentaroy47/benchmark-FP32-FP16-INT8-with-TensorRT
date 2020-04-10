# benchmark-FP32-FP16-INT8-in-pytorch
Benchmark inference speed of CNNs with various quantization methods in Pytorch!

# Image classification

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
| INT8 | 1.7      | 2.7      | 3.9      |

## FP32 vs FP16
`inference_FP32_vs_FP16.ipynb`

Benchmarks inference speed with FP32 and FP16 with amp.

