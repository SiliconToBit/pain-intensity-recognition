# Pain Intensity Recognition — ResNet-18 + LSTM

基于 ResNet-18 + LSTM 的疼痛强度识别模型，在 MIntPAIN 数据集上进行 5 级疼痛分类。

---

## 模型架构

### ResNet-18 + LSTM

端到端训练，无需中间特征提取步骤：

```
输入: (B, T, C, H, W) — T帧图像序列
  ↓ reshape to (B*T, C, H, W)
ResNet-18 (ImageNet 预训练) → (B*T, 512)
  ↓ reshape to (B, T, 512)
LSTM → (B, hidden_dim)
  ↓
FC → (B, num_classes)
```

- **骨干网络**: ResNet-18，ImageNet 预训练
- **时序建模**: LSTM (hidden_dim=256, 1 layer)
- **分类头**: Dropout + Linear(256→5)

### 训练策略（分阶段）

| 阶段 | Backbone | 分类器 | Epochs | LR |
|------|----------|--------|--------|----|
| Phase 1 | 冻结 | 训练 | 10 | 5e-4 |
| Phase 2 | 微调 (5e-5) | 训练 (5e-4) | 20 | 分组学习率 + warmup |

**Phase 2 Warmup**: 前 3 个 epoch 线性提升 backbone LR 从 0 → 5e-5，稳定微调初期

- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau (monitor val loss)
- **早停**: patience=5
- **数据增强**: RandomHorizontalFlip, RandomAffine(±10°, ±5%平移)

### 验证

- **LOSO 交叉验证**: 20 折留一受试者验证

---

## 评估指标

| 指标 | 说明 |
|------|------|
| **Weighted F1-score** | 对类别不平衡敏感，按样本数加权 |
| **Macro F1-score** | 各类别同等权重 |
| **Per-class Recall** | 每个疼痛等级的召回率 |
| **Confusion Matrix** | 分析相邻等级的混淆情况 |
| **Cohen's Kappa** | 校正偶然一致性 |
| **Multi-class AUROC** | 整体区分能力 (OvR) |

---

## 项目结构

```
├── README.md               # 使用说明
├── requirements.txt        # 依赖清单
├── main.py                 # 主入口
├── config.py               # 配置参数
├── model.py                # ResNet-18 + LSTM 模型
├── train.py                # 训练、评估、指标计算
└── utils/
    ├── dataset.py           # 图像序列数据集
    └── checkpoint.py        # 断点续训工具
```

---

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 完整 LOSO 交叉验证
python main.py

# 快速验证（1-2 折）
python main.py --num_folds 1

# 断点续训
python main.py --resume

# 使用配置文件
python main.py --config my_config.yaml
```

---

## 配置参数 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_classes | 5 | 疼痛等级 |
| sequence_length | 5 | 时序窗口长度 |
| backbone | resnet18 | 骨干网络 |
| pretrained | True | 使用 ImageNet 预训练 |
| phase1_epochs | 20 | Phase 1 训练轮数 |
| phase2_epochs | 10 | Phase 2 训练轮数 |
| phase1_lr | 1e-3 | Phase 1 学习率 |
| phase2_backbone_lr | 1e-4 | Phase 2 backbone 学习率 |
| phase2_classifier_lr | 1e-3 | Phase 2 分类器学习率 |
| batch_size | 32 | 批量大小 |
| patience | 5 | 早停耐心值 |
| lstm_hidden_dim | 256 | LSTM 隐藏维度 |
| lstm_num_layers | 1 | LSTM 层数 |
| dropout | 0.5 | Dropout 比率 |
| undersample | True | 训练欠采样平衡 |
| class_weight | inverse | 类别加权 (none/inverse/sqrt_inverse) |
| num_folds | 0 (全部) | LOSO 折数 |

---

## 输出

训练完成后，结果保存在 `<output_dir>/`:

```
├── results.json           # 所有指标汇总
├── predictions.npy        # 预测结果
├── labels.npy             # 真实标签
├── probabilities.npy      # 预测概率
├── confusion_matrix.npy   # 混淆矩阵
└── checkpoints/           # 模型 checkpoint
```

---

## Checkpoint 断点续训

```
<output_dir>/checkpoints/
├── fold00_latest.pth      # fold 0 最新 checkpoint
├── fold00_epoch001.pth    # fold 0 第 1 个 epoch
├── train_progress.json    # 已完成的 folds
└── ...
```
