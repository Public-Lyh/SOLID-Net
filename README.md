SOLID-Net

#中文说明 | #english-description

中文说明

1. 项目简介

SOLID-Net 是一个用于手部康复手势识别任务的深度学习模型。部分核心代码将在论文被接收后开源。

本仓库包含：

• 模型训练代码；

• 对比实验代码；

• 消融实验代码；

• 数据集下载方式；

• 预训练模型下载方式。

2. 仓库结构

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
└── README.md


3. 环境配置

建议使用 Python 版本：
Python == 3.11.14


安装依赖（依赖文件暂未补全）：
pip install -r requirements.txt


如果使用 GPU，请根据您的 CUDA 版本安装合适的 PyTorch。

PyTorch 官网：

<https://pytorch.org/get-started/locally/>

4. 数据集

由于自建的 SeeEMG 数据集较大，未直接上传至 GitHub。

数据集下载地址：暂未开放
/yourpath/Dataset


推荐目录结构：
data/
├── train/
├── val/
└── test/


5. 预训练模型

由于模型文件较大，未直接上传至 GitHub。

预训练模型下载地址暂未开放。

6. 模型训练

配置好数据集路径和 Python 环境后，在 VSCode 中打开项目并运行相应脚本即可开始训练。

7. 引用

如果您觉得本项目有帮助，请引用：

（待补充）


English Description

1. Introduction

SOLID-Net is a deep learning model for hand rehabilitation gesture recognition. Some core code will be open-sourced after the paper is accepted.

This repository contains:

• Training code;

• Comparison experiment code;

• Ablation experiment code;

• Dataset download instructions;

• Pretrained model download instructions.

2. Repository Structure

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
└── README.md


3. Environment Setup

Recommended Python version:
Python == 3.11.14


Install dependencies:
pip install -r requirements.txt


If you use a GPU, please install the appropriate PyTorch version according to your CUDA version.

PyTorch official website:

<https://pytorch.org/get-started/locally/>

4. Dataset

The dataset is too large to be uploaded directly to GitHub.

Dataset download link will be provided soon.

After downloading, please place the dataset under:
/yourpath/Dataset


Recommended dataset structure:
data/
├── train/
├── val/
└── test/


5. Pretrained Models

The pretrained model weights are not uploaded to GitHub due to their large size.

Download links will be provided soon.

6. Training

After configuring the dataset path and Python environment, you can start training by opening the project in VSCode and running the desired script.

7. Citation

If you find this project helpful, please cite:

(to be added)
