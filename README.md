# SOLID-Net

[中文](#中文说明) | [English](#english-description)

---

# 中文说明

## 1. 项目简介

SOLID-Net 是一个面向手部康复手势识别任务的深度学习模型。部分核心代码将在论文被接收后开源。

本仓库包含：

- 模型训练代码；
- 对比实验代码；
- 消融实验代码；
- 数据集获取方式；
- 预训练模型获取方式。

## 2. 仓库结构

text
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


## 3. 环境配置

推荐 Python 版本：

text
Python == 3.11.14


安装依赖（依赖文件暂未提供）：

bash
pip install -r requirements.txt


如果使用 GPU，请根据您的 CUDA 版本安装相应的 PyTorch。

PyTorch 官方网站：

<https://pytorch.org/get-started/locally/>

## 4. 数据集

由于自建的 SeeEMG 数据集体积较大，未直接上传至 GitHub。

数据集下载地址：暂未开放。

下载后，请将数据集放置在以下路径：

text
/yourpath/Dataset


推荐目录结构：

text
data/
├── train/
├── val/
└── test/


## 5. 预训练模型

由于模型文件较大，未直接上传至 GitHub。

预训练模型下载地址：暂未开放。

## 6. 模型训练

配置好数据集路径和 Python 环境后，在 VSCode 中直接运行相应脚本即可开始训练。

## 7. 引用

如果您觉得本项目有帮助，请引用：




---

# English Description

## 1. Introduction

SOLID-Net is a deep learning model designed for hand rehabilitation gesture recognition. Some core code will be open-sourced after the paper is accepted.

This repository contains:

- Training code;
- Comparison experiment code;
- Ablation experiment code;
- Dataset download instructions;
- Pretrained model download instructions.

## 2. Repository Structure

text
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


## 3. Environment Setup

Recommended Python version:

text
Python == 3.11.14


Install dependencies:

bash
pip install -r requirements.txt


If you use a GPU, please install the appropriate PyTorch version based on your CUDA version.

PyTorch official website:

<https://pytorch.org/get-started/locally/>

## 4. Dataset

Due to its large size, the custom SeeEMG dataset is not directly uploaded to GitHub.

Dataset download link: coming soon.

After downloading, please place the dataset under the following path:

text
/yourpath/Dataset


Recommended dataset structure:

text
data/
├── train/
├── val/
└── test/


## 5. Pretrained Models

Due to their large size, the pretrained model weights are not uploaded to GitHub.

Download links: coming soon.

## 6. Training

After configuring the dataset path and Python environment, you can start training by running the corresponding scripts directly in VSCode.

## 7. Citation

If you find this project helpful, please cite:

