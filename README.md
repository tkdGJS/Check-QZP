# Check-QZP

Check-QZP is a lightweight checkpointing mechanism that reduces checkpoint delays and storage requirements during deep learning training. When multiple workloads share a storage device, I/O contention can delay checkpoint writes and leave the GPU idle.

## How it works

- **Q · Quantization** converts 32-bit floating-point checkpoint values to 8-bit representations to reduce the amount of data written.
- **Z · Compression** uses gzip to further reduce the size of the quantized checkpoint.
- **P · Parallelism** overlaps checkpoint processing on the CPU with training on the GPU to reduce stalls.

## Results

The paper evaluated Check-QZP in PyTorch on VisDrone object detection and tracking workloads, using fio to reproduce storage I/O contention. Compared with baseline checkpointing, Check-QZP reduced **checkpoint time by up to 87.5%**, **the amount of data written by up to 99.8%**, and **total training time by up to 1.5%**.

This repository contains [object detection experiments](drone/), [multi-object tracking experiments](MOT/), and [fio workload scripts](fio_test/).

[Check-QZP: A Lightweight Checkpoint Mechanism for Deep Learning Frameworks](https://doi.org/10.3390/app14198848) · *Applied Sciences*, 2024
