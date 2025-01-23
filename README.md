# PolaFormer: Polarity-aware Linear Attention for Vision Transformers

This repo contains the official **PyTorch** code and pre-trained models for PolaFormer (ICLR 2025). [[Paper Link]](https://openreview.net/pdf?id=kN6MFmKUSK)

## Introduction

### Motivation

Linear attention has emerged as a promising alternative to softmax-based attention, leveraging kernelized feature maps to reduce complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ in sequence length. However, the non-negative constraint on feature maps and the relaxed exponential function used in approximation lead to significant information loss compared to the original query-key dot products, resulting in less discriminative attention maps with higher entropy.
 To address the missing interactions driven by negative values in query-key pairs and the high entropy, we propose the **PolaFormer**, which achieves a superior balance between expressive capability and efficiency.

### Method

 In this paper,  we propose the polarity-aware linear attention mechanism that explicitly models both same-signed and opposite-signed query-key interactions, ensuring comprehensive coverage of relational information. Furthermore, to restore the spiky properties of attention maps, we prove that the existence of a class of element-wise functions (with positive first and second derivatives) can reduce entropy in the attention distribution. Finally, we employ a learnable power function for rescaling, allowing strong and weak attention signals to be effectively separated. 
 <p align="center">
    <img src="figures/mainfig.jpg" width= "600">
</p>
 Notably, we introduce two learnable polarity-aware coefficients matrices applied with element-wise multiplication, which are expected to learn the complementary relationship between same-signed and opposite-signed values.  
<p align="center">
    <img src="figures/PearsonPCA.jpg" width= "500">
</p>

### Results

- Comparison of different models on ImageNet-1K.

<p align="center">
    <img src="figures/pola.png" width= 500>
</p>

- Performance on Long Range Arena benchmark.

| Model                 | Text  | ListOps | Retrieval      | Pathfinder | Image | Average |
|-----------------------|-------|---------|----------------|------------|-------|---------|
| $\text{PolaFormer}_{\alpha=3}$ | 73.06 | 37.35   | 80.50          | 70.53      | 42.15 | 60.72   |
| $\text{PolaFormer}_{\alpha=5}$ | 72.33 | 38.76   | 80.37          | 68.98      | 41.91 | 60.47   |
| $\text{PolaFormer}_{\alpha=7}$ | 71.93 | 37.60   | 81.47          | 69.09      | 42.77 | 60.57   |




## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Pretrained Models

Based on different model architectures, we provide several pretrained models, as listed below.

| model  | Reso | acc@1 | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: |
| FLatten-PVT-T | $224^2$ | 78.8 (+3.7) | [config](cfgs/pola_pvt_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3ab1d773f19d45648690/?dl=1) |
| FLatten-PVT-S | $224^2$ | 81.9 (+2.1) | [config](cfgs/pola_pvt_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/5d1f01532b104da28e7b/?dl=1) |
| FLatten-Swin-T | $224^2$ | 82.6 (+1.4) | [config](cfgs/pola_swin_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e1518e76703e4e57a7f2/?dl=1) |
| FLatten-Swin-S | $224^2$ | 83.6 (+0.6) | [config](cfgs/pola_swin_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/94188e52af354bf4a88b/?dl=1) |
| FLatten-Swin-B | $224^2$ | 83.8 (+0.3) | [config](cfgs/pola_swin_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/7a9e5186bad04e7fb3a9/?dl=1) |


Evaluate one model on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

## Train Models from Scratch

- **To train our model on ImageNet from scratch, see pretrain.sh and run:**

```shell
bash pretrain.sh
```

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [FLatten Transformer](https://github.com/LeapLabTHU/FLatten-Transformer).
## Citation

If you find this repo helpful, please consider citing us.

```latex
@InProceedings{han2023flatten,
  title={FLatten Transformer: Vision Transformer using Focused Linear Attention},
  author={Han, Dongchen and Pan, Xuran and Han, Yizeng and Song, Shiji and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Contact

If you have any questions, please feel free to contact the authors. 

Weikang Meng: [zacharymengwk@gmail.com](mailto:zacharymengwk@gmail.com)
