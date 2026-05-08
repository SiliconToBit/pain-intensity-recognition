# EDLM MIntPAIN Reproduce

本项目用于复现 EDLM (Ensemble Deep Learning Model) 在 MIntPAIN 数据集上的疼痛等级分类实验。

## 算法流程

1. **早期融合**: 微调 VGGFace2 (VGG-16 架构) 作为特征提取器，输出 4 维特征向量
2. **PCA 降维**: 将 4 维特征降至 3 维（训练集拟合，测试集变换，无泄露）
3. **时序建模**: 滑动窗口 5 帧，步长 5 帧，生成 (5, 3) 序列
4. **欠采样**: 训练阶段对各类别欠采样至平衡，缓解无痛帧过多问题
5. **三流集成网络**: DNN1 (Conv1D+BiLSTM)、DNN2 (Conv1D+BiLSTM)、DNN3 (Conv1D+LSTM) 合并输出
6. **LOSO 交叉验证**: 20 折留一受试者验证，输出 Accuracy、AUC、F1、混淆矩阵

## 项目结构

```
edlm_mintpain_reproduce/
├── README.md                  # 使用说明
├── requirements.txt           # 依赖清单
├── main.py                    # 一键运行主脚本
├── config.py                  # 所有可配置参数
├── model.py                   # FeatureExtractor (VGGFace2) + 三流 EnsembleEDLM
├── feature_extraction.py      # VGG微调、4D特征提取、PCA降维、5帧窗口生成
├── train.py                   # LOSO交叉验证训练与评估（含梯度裁剪/早停）
└── utils/
    ├── dataset.py             # 时序数据集加载
    ├── download_utils.py      # VGGFace2 权重下载指引
    └── face_alignment.py      # 人脸检测与对齐（备用）
```

## 数据集

数据集位于 `/home/gm/dataset/mintpain`，包含已预处理的 224x224 人脸图像和 LOSO 划分文件。

## VGGFace2 权重（强烈推荐）

论文使用 VGGFace（在人脸数据集上预训练），而非 ImageNet。使用 VGGFace2 权重可显著提升表情特征敏感度：

```bash
# 下载转换后的 PyTorch VGGFace2 权重
mkdir -p /home/gm/dataset/mintpain/weights
# 推荐来源: https://github.com/ox-vgg/vgg_face2 或 https://github.com/cydonia999/VGGFace2-Pytorch
# 将权重文件重命名为 vgg_face2.pth 并放置于 weights 目录
```

如未提供 VGGFace2 权重，代码将自动回退到 ImageNet 预训练权重。

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 运行完整流程（特征提取 + 训练评估）
python main.py

# 仅运行训练（跳过特征提取）
python main.py --skip_extraction

# 仅运行特征提取
python main.py --skip_train
```

## 关键参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| mintpain_root | /home/gm/dataset/mintpain | 数据集根目录 |
| num_classes | 5 | 疼痛等级分类数 |
| sequence_length | 5 | 时序窗口长度（帧） |
| bottleneck_dim | 4 | VGG 特征瓶颈维度 |
| pca_dim | 3 | PCA 降维后维度 |
| undersample | True | 训练时类别欠采样平衡 |
| feature_extractor_epochs | 50 | 特征提取器微调 epoch 数 |
| ensemble_epochs | 5 | 集成模型训练 epoch 数 |
| batch_size | 48 | 批量大小 |
| learning_rate | 1e-4 | 学习率 |
| num_folds | 20 | LOSO 折数（=受试者数） |

## 稳健性增强

- **数据增强**: 微调时随机水平翻转、小角度仿射变换
- **学习率调度**: StepLR (每 20 epoch 衰减 0.5)
- **梯度裁剪**: RNN 训练时 max_norm=1.0
- **早期停止**: 特征提取器 (patience=5) 和集成模型 (patience=3) 均监控验证损失
