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

|      | resnet18          | resnet34          | resnet50          |
|------|-------------------|-------------------|-------------------|
| Raw  | 0.010930915573734 | 0.012351334394522 | 0.016015976517644 |
| FP32 | 0.003857297513952 | 0.00567102072826  | 0.009893084291238 |
| FP16 | 0.002160526400235 | 0.003377080562726 | 0.004446487330911 |
| INT8 | 0.001726765129434 | 0.002702511734699 | 0.003906130191669 |

## FP32 vs FP16
`inference_FP32_vs_FP16.ipynb`

Benchmarks inference speed with FP32 and FP16 with amp.

