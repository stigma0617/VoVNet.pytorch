# A pytorch Implementation of VoVNet Backbone Networks

This is a pytorch implementation of VoVNet backbone networks as described in the paper [An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection](https://arxiv.org/abs/1904.09730). This is implemented through [pytorch/vision](https://github.com/pytorch/vision/tree/master/torchvision/models) style.

## What does this repo provide?  
This repo provides VoVNet-39/57 models trained on ImageNet classification dataset with same training protocols as [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet) (e.g., 128 batch size, 90 epoch, data augmentations, lr_policy, etc) and compares to ResNet and DenseNet.


## ImageNet results

### Notes:
 - For fair comparison, same training protocols used as [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
    - 90 epoch
    - step learning rate schedule (learning rate decayed at every 30 epoch)
    - 256 batch size
    - default augmentations (e.g., crop, flip, same mean/std normalization)
    - @224x224 training/validation
 - Inference time is measured on TITAN X PASCAL GPU
    - CUDA v9.2, cuDNN v7.3, pytorch 1.0
 

| Model | Top-1 | Top-5 | Inference time
| :--:  |  :--: | :--:  | :--: |
| ResNet-50     |  23.85%     | 7.13%     |12 ms|
| DenseNet-201  |  22.80%     | 6.43%     |39 ms|
| **VoVNet-39** |  23.23%     | 6.57%     |**10** ms|
| ResNet-101    |  22.63%     | 6.44%     |20 ms|
| DenseNet-161  |  22.35%     | 6.20%     |27 ms|
| **VoVNet-57** |  **22.27**% | **6.28**% |13 ms |



## Pretrained models

- [VoVNet-39](https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1)
- [VoVNet-57](https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1)


## Training & Inferecne

### Installation
0. python  3.6
1. Install pytorch > 0.4
2. `git clone https://github.com/stigma0617/VoVNet.pytorch.git`
3. install depenecies : `pip install -r requirements.txt`

### Imagenet data preparation

You can follow the instruction of [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)


### Train

run `main.py` specifying data path, the desired model name`--arch`, and save-directory`--savedir`.


````bash
python main.py [imagenet-folder with train and val folders] --arch vovnet39 --savedir VoVNet39
````


### Related projects

[VoVNet-Detectron](https://github.com/stigma0617/maskrcnn-benchmark-vovnet/tree/vovnet)
[VoVNet-DeepLabV3](https://github.com/stigma0617/VoVNet-DeepLabV3)


### TO DO

 - [ ] VoVNet-27-slim
 - [ ] VoVNet-27-slim-depthwise
 - [ ] VoVNet-99
