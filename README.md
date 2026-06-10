# SOLID-Net

[中文](#中文说明) | [English](#english-description)

---

# 中文说明

## 1. 项目简介

SOLID-Net 是一个用于手部康复手势识别任务的深度学习模型。

本仓库包含：

- 模型训练代码；
- 对比实验代码；
- 消融实验代码；
- 数据集下载方式；
- 预训练模型下载方式。

## 2. 仓库结构

```text
SOLID-Net/
├── training/
│   ├── Fusion Model.py
│   ├── vision.py
│   └── emg.py
├── comparison_experiments/
│   ├── Comparison-Emg.py
│   ├── Comparison-Vis.py
│   └── Comparison-Fusion.py
├── ablation_experiments/
│   ├── Ablation-Emg.py
│   ├── Ablation-Vis.py
│   └── Ablation-Fusion.py
├── requirements.txt
└── README.md
```

## 3. 环境配置

建议使用 Python 版本：

```text
Python == 3.11.14
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 GPU，请根据自己的 CUDA 版本安装合适的 PyTorch。

PyTorch 官网：

<https://pytorch.org/get-started/locally/>

## 4. 数据集

由于自建的SeeEMG数据集较大，未直接上传至 GitHub。

数据集下载地址：

- 百度网盘：你的链接
- 提取码：xxxx

下载后，请将数据集放置在：

```text
/yourpath/Dataset
```

推荐目录结构：

```text
data/
├── train/
├── val/
└── test/
```

## 5. 预训练模型

由于模型文件较大，未直接上传至 GitHub。

预训练模型下载地址：

- 百度网盘：你的链接
- 提取码：xxxx
- Google Drive：你的链接，可选

下载后，请将模型权重放置在：

```text
checkpoints/
```

## 6. 模型训练

配置好数据集路径和python环境后在VSCode右键开始运行即可开始训练。

## 7.引用

如果你觉得本项目有帮助，请引用：

```bibtex
@article{yourpaper2025solidnet,
  title={SOLID-Net: Your Paper Title},
  author={Your Name},
  journal={Your Journal or Conference},
  year={2025}
}
```

## 8. 联系方式

如有问题，请联系：

- GitHub Issues: https://github.com/Public-Lyh/SOLID-Net/issues
- Email: luoyuhang963@gmail.com 、 2586160590@qq.com

---

# English Description

## 1. Introduction

SOLID-Net is a deep learning model designed for XXX.

This repository contains:

- Training code;
- Comparison experiment code;
- Ablation experiment code;
- Dataset download instructions;
- Pretrained model download instructions.

## 2. Repository Structure

```text
SOLID-Net/
├── training/
│   ├── Fusion Model.py
│   ├── vision.py
│   └── emg.py
├── comparison_experiments/
│   ├── Comparison-Emg.py
│   ├── Comparison-Vis.py
│   └── Comparison-Fusion.py
├── ablation_experiments/
│   ├── Ablation-Emg.py
│   ├── Ablation-Vis.py
│   └── Ablation-Fusion.py
├── requirements.txt
└── README.md
```

## 3. Environment Setup

Recommended Python version:

```text
Python == 3.11.14
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use GPU, please install the appropriate PyTorch version according to your CUDA version.

PyTorch official website:

<https://pytorch.org/get-started/locally/>

## 4. Dataset

The dataset is not uploaded to GitHub due to its large size.

Dataset download link:

- Baidu Netdisk: your link
- Extraction code: xxxx

After downloading, please place the dataset under:

```text
data/
```

Recommended dataset structure:

```text
data/
├── train/
├── val/
└── test/
```

## 5. Pretrained Models

The pretrained model weights are not uploaded to GitHub due to their large size.

Download links:

- Baidu Netdisk: your link
- Extraction code: xxxx
- Google Drive: your link, optional

After downloading, please place the weights under:

```text
checkpoints/
```

## 6. Training

After configuring the dataset path and Python environment, you can start training by right-clicking and running in VSCode.

## 7. Citation

If you find this project helpful, please cite:

```bibtex
@article{yourpaper2025solidnet,
  title={SOLID-Net: Your Paper Title},
  author={Your Name},
  journal={Your Journal or Conference},
  year={2025}
}
```

## 11. Contact

If you have any questions, please contact:

- GitHub Issues: https://github.com/Public-Lyh/SOLID-Net/issues
- Email: luoyuhang963@gmail.com 、 2586160590@qq.com
