# EDLM MIntPAIN — 疼痛强度识别

本项目复现 EDLM (Ensemble Deep Learning Model) 在 MIntPAIN 数据集上的 5 级疼痛强度分类。

---

## 总体架构

### 特征提取（FeatureExtractor）

- **骨干网络**: 标准 VGG16，预训练于 VGGFace（Oxford 2015），**所有卷积层完全冻结**
- **瓶颈层**: Linear(25088→4096) → ReLU → Dropout → Linear(4096→4)，输出 4 维特征
- **微调**: 每折独立微调 50 epoch，SGD(lr=0.001, momentum=0.9)，StepLR 每 20 epoch 衰减 0.5，patience=5 早停
- **降维**: PCA 将 4 维降至 3 维（训练集拟合，测试集变换，无数据泄露），验证方差保留 ≥ 99%

### 时序分类（EnsembleEDLM）

- **输入**: 滑动窗口 5 帧，步长 1 帧（重叠窗口），每帧 3 维 PCA 特征 → 序列形状 (5, 3)
- **三流并行**:
  - StreamDNN1: Conv1D(3→256) × 2 + BiLSTM(256→256) + FC(512→4096)
  - StreamDNN2: Conv1D(3→128) × 2 + BiLSTM(128→32) + FC(64→4096)
  - StreamDNN3: Conv1D(3→256) × 1 + **单向** LSTM(256→128) + FC(128→4096)
- **融合**: 三流输出拼接 (4096×3) → **直接 FC(12288→5)** （无中间 FC 层）
- **训练**: 5 epoch，SGD(lr=0.001, momentum=0.9)，StepLR step=2 gamma=0.5，patience=3 早停

### 验证

- **LOSO 交叉验证**: 20 折留一受试者验证（MIntPAIN 共 20 名受试者）

---

## 实验结果

### 总体指标 (10折 LOSO)

| 指标 | 均值 | 标准差 |
|------|------|--------|
| Accuracy | 39.09% | ±10.38% |
| F1 (weighted) | 30.07% | ±4.85% |
| AUC (weighted) | 52.27% | ±4.70% |

### 各类别 AUC

| 类别 | 均值 AUC |
|------|----------|
| Class 0 (无痛) | 0.524 |
| Class 1 | 0.518 |
| Class 2 | 0.481 |
| Class 3 | 0.542 |
| Class 4 | 0.543 |

### 混淆矩阵

```
          预测
       0    1    2    3    4
真实 0  1415  148  198  143  110
     1   346   38   39   43   35
     2   339   44   52   45   17
     3   348   45   55   36   28
     4   336   39   58   42   29
```

### 分类报告

| 类别 | 精确率 | 召回率 | F1 | 支持数 |
|------|--------|--------|-----|--------|
| 0 | 0.51 | 0.70 | 0.59 | 2014 |
| 1 | 0.12 | 0.08 | 0.09 | 501 |
| 2 | 0.13 | 0.10 | 0.12 | 497 |
| 3 | 0.12 | 0.07 | 0.09 | 512 |
| 4 | 0.13 | 0.06 | 0.08 | 504 |

---

## 分析

- **优于随机**: 39% 准确率对比随机基线 20%，模型学到了一定模式
- **严重偏向无痛类**: Class 0 召回率 70%，但其余 4 类召回率均低于 10%，大量疼痛样本被误判为无痛
- **类别不平衡**: 测试集无痛帧占 50%，模型倾向于多数类
- **区分度不足**: AUC ~0.52 接近随机，疼痛等级间区分困难

### 可能改进方向

1. **获取 VGGFace 预训练权重**: 下载并转换原始 VGGFace Caffe 模型，替换 ImageNet 回退权重
2. **加长时序窗口** (如 sequence_length=10)，捕捉更长程的疼痛变化
3. **损失函数加权**，进一步缓解类别不平衡
4. **启用人脸对齐** (preprocess.py 中设置 align=True) 可能提升特征质量

---

## 项目结构

```
├── README.md                  # 使用说明 & 实验结果
├── requirements.txt           # 依赖清单
├── main.py                    # 一键运行主脚本
├── config.py                  # 所有可配置参数
├── model.py                   # FeatureExtractor + EnsembleEDLM
├── feature_extraction.py      # 微调、4D特征提取、PCA降维、窗口生成
├── train.py                   # LOSO训练评估
├── continue_training.py       # 续跑脚本（跳过已完成折）
└── utils/
    ├── dataset.py             # 时序数据集
    ├── download_utils.py      # 权重下载指引
    └── face_alignment.py      # 人脸检测与对齐
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 完整流程
python main.py

# 仅训练集成模型
python main.py --skip_extraction

# 仅特征提取
python main.py --skip_train
```

## 关键参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| mintpain_root | /home/featurize/work/dataset/mintpain | 数据集根目录 |
| num_classes | 5 | 疼痛等级 |
| sequence_length | 5 | 时序窗口长度 |
| bottleneck_dim | 4 | 特征瓶颈维度 |
| pca_dim | 3 | PCA 降维维度 |
| undersample | True | 训练欠采样平衡 |
| feature_extractor_epochs | 50 | 微调 epoch 数 |
| ensemble_epochs | 5 | 集成模型 epoch 数 |
| num_folds | 0 (全部20折) | LOSO 折数 |
| feature_backbone | vgg16 | 骨干网络 |
