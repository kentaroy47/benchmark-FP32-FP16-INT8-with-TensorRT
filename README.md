# benchmark-FP32-FP16-INT8-in-pytorch
Benchmark inference speed of CNNs with various quantization methods in Pytorch!

Hardware:Jetson Nano.

Latency of image inference (1,3,256,256) [ms]

|          | FP32 | FP16 | INT8 |
|:--------:|------|:----:|------|
| resnet18 | 26   |  18  |      |
| resnet34 | 48   |  30  |      |
| resnet50 | 79   | 42   |      |

Jetson Nano does not support INT8..

## FP32 vs FP16
`inference_FP32_vs_FP16.ipynb`

Benchmarks inference speed with FP32 and FP16 with amp.

