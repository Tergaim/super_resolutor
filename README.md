# SuperResolutor

## Description

SuperResolutor is a collection of pytorch models for super-resolution, i.e. computing a high-resolution image from a low-resolution one.
By default, output size is 64*64 and scale factor is 2. Unless stated otherwise, models work in B&W, but they can be adapted to work colors easily.

## Implemented models

### Pre-Upscaled models
These models take an input of the same size as the output. In this case, `cv2.resize` is used to transform the input image with bilinear interpolation.

1. [SRCNN](https://arxiv.org/abs/1501.00092) is to my knowledge the first model for super-resolution using deep learning and is comprised of three convolution layers. Trained with MSE loss.
2. [VDSR](https://arxiv.org/abs/1511.04587), inspired by the VCC architecture, is a tower of 3x3 convolutions. The paper also proposes a specific training method, which is not implemented (yet?).

### Post_Upscaled models

Here, the upscale is done by the model itself, usually at the end of computation. This reduces the computational effort required with generally better performances.
1. [FSRCNN](https://arxiv.org/abs/1608.00367)