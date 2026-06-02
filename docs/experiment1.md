# 实验一：ResNet-18 + LSTM 疼痛强度识别

## 1. 实验目标

基于 MIntPAIN 数据集，使用 ResNet-18 作为视觉特征提取器、LSTM 建模时序关系，实现 5 级疼痛强度分类。重点关注类别不平衡问题的处理和模型泛化能力评估。

## 2. 模型架构

### 2.1 整体结构

```
输入: (B, T, C, H, W) — 批次 × T帧 × 3通道 × 224 × 224
  │
  ├─ reshape → (B×T, 3, 224, 224)
  │
  ├─ ResNet-18 (ImageNet 预训练)
  │   ├─ Conv1 + BN + ReLU + MaxPool
  │   ├─ Layer1 (2× BasicBlock, 64-dim)
  │   ├─ Layer2 (2× BasicBlock, 128-dim)
  │   ├─ Layer3 (2× BasicBlock, 256-dim)
  │   ├─ Layer4 (2× BasicBlock, 512-dim)
  │   └─ AdaptiveAvgPool → 512-dim
  │
  ├─ reshape → (B, T, 512)
  │
  ├─ LSTM (input=512, hidden=256, layers=1)
  │   └─ 取最后时步输出 → (B, 256)
  │
  └─ Classifier
      ├─ Dropout(0.5)
      └─ Linear(256 → 5) → logits
```

### 2.2 各层参数量

| 模块 | 输出维度 | 参数量 | 是否可训练 |
|------|----------|--------|------------|
| ResNet-18 Backbone | 512 | 11,176,512 | Phase 1 冻结 / Phase 2 微调 |
| LSTM | 256 | 787,456 | 始终可训练 |
| Dropout + Linear | 5 | 1,285 | 始终可训练 |
| **总计** | — | **11,965,253** | — |

### 2.3 输入处理

- **图像尺寸：** 224 × 224 (已在预处理阶段裁剪)
- **归一化：** ImageNet 标准 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **时序窗口：** 5 帧滑动窗口，步长 1 帧 (重叠采样)

## 3. 评估指标

采用 6 项综合评估指标，全面衡量模型性能：

### 3.1 Weighted F1-score

$$F1_{weighted} = \sum_{i=0}^{K-1} w_i \cdot F1_i, \quad w_i = \frac{n_i}{N}$$

- 按各类别样本数加权，对类别不平衡敏感
- 作为主要优化目标

### 3.2 Macro F1-score

$$F1_{macro} = \frac{1}{K} \sum_{i=0}^{K-1} F1_i$$

- 各类别等权重，不考虑样本数差异
- 衡量模型对少数类的识别能力

### 3.3 Per-class Recall (每类召回率)

$$Recall_i = \frac{TP_i}{TP_i + FN_i}$$

- 逐类别计算，观察模型对每个疼痛等级的识别能力
- 特别关注疼痛类 (Label 1-4) 的召回率

### 3.4 Confusion Matrix (混淆矩阵)

- K×K 矩阵，行=真实标签，列=预测标签
- 分析相邻等级的混淆模式
- 观察是否存在系统性偏差 (如偏向多数类)

### 3.5 Cohen's Kappa

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

- $p_o$：观测一致率
- $p_e$：期望一致率 (偶然一致性)
- 校正随机猜测的影响，更真实反映模型性能
- 解读：0-0.2 差, 0.2-0.4 一般, 0.4-0.6 中等, 0.6-0.8 好, >0.8 优秀

### 3.6 Multi-class AUROC

$$AUROC_{weighted} = \sum_{i=0}^{K-1} w_i \cdot AUROC_i$$

- One-vs-Rest 策略，计算每个类别的一对多 AUROC
- 衡量模型的整体区分能力
- 对类别不平衡有一定鲁棒性

## 4. 优化策略

### 4.1 类别不平衡处理 (双重策略)

#### 策略一：欠采样 (Undersampling)

```
原始分布:                    欠采样后:
Label 0: 1561 (50.0%)  →   Label 0: 388 (20.0%)
Label 1:  388 (12.4%)  →   Label 1: 388 (20.0%)
Label 2:  392 (12.5%)  →   Label 2: 388 (20.0%)
Label 3:  388 (12.4%)  →   Label 3: 388 (20.0%)
Label 4:  393 (12.6%)  →   Label 4: 388 (20.0%)
```

- 多数类随机下采样至与少数类相同数量
- 避免模型过度偏向无痛类
- 代价：丢弃部分多数类数据

#### 策略二：加权损失函数 (Weighted CrossEntropyLoss)

$$\mathcal{L} = -\sum_{i=0}^{K-1} w_i \cdot y_i \log(\hat{y}_i)$$

$$w_i = \frac{N}{n_i \times K}$$

| 类别 | 样本数 | 权重 |
|------|--------|------|
| Label 0 | 1561 | 0.32 |
| Label 1 | 388 | 1.29 |
| Label 2 | 392 | 1.28 |
| Label 3 | 388 | 1.29 |
| Label 4 | 393 | 1.28 |

- 少数类的错误预测惩罚更大
- 与欠采样互补：欠采样平衡数据分布，加权损失平衡梯度贡献

### 4.2 迁移学习策略 (分阶段训练)

#### Phase 1：冻结 Backbone (10 epochs)

```
Backbone (ResNet-18):  冻结 ❄️
LSTM + Classifier:     训练 🔥
学习率: 5e-4
```

- 仅训练新初始化的 LSTM 和分类头
- 利用 ImageNet 预训练的视觉特征
- 快速收敛，为 Phase 2 提供稳定的初始化

#### Phase 2：微调 Backbone (20 epochs)

```
Backbone (ResNet-18):  微调 🔥 (lr=5e-5)
LSTM + Classifier:     训练 🔥 (lr=5e-4)
Warmup: 前 3 epochs 线性提升 backbone LR 0 → 5e-5
```

- 解冻 backbone，使用较小学习率微调
- Warmup 避免解冻初期梯度震荡破坏预训练特征
- 分组学习率：backbone (5e-5) 是 classifier (5e-4) 的 1/10

**学习率变化曲线：**

```
LR
│
5e-4 ──────────────────────────────────────── Classifier LR
│
5e-5 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  Backbone LR 目标
│           ╱
│         ╱    Warmup (3 epochs)
│       ╱
0  ────╱────────────────────────────────────→ Epoch
     Phase 1    Phase 2
     (10 ep)    (20 ep)
```

### 4.3 数据增强

| 增强方法 | 参数 | 作用 |
|----------|------|------|
| RandomResizedCrop | scale=(0.8, 1.0) | 随机裁剪后缩放，增加尺度变化 |
| RandomHorizontalFlip | p=0.5 | 水平翻转，增加样本多样性 |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 | 模拟光照变化 |
| RandomAffine | degrees=10, translate=5% | 旋转和平移，增加几何变化 |
| Normalize | ImageNet mean/std | 与预训练分布对齐 |

**注意：** 测试集不使用任何增强，仅做 Resize + Normalize。

### 4.4 正则化

| 方法 | 参数 | 位置 |
|------|------|------|
| Dropout | p=0.5 | LSTM 输出后、Linear 前 |
| Early Stopping | patience=5 | 监控验证集 loss |
| LR Scheduler | ReduceLROnPlateau | factor=0.5, patience=2 |

### 4.5 训练超参数

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| Batch Size | 32 |
| 序列长度 (T) | 5 帧 |
| 滑动窗口步长 | 1 帧 |
| LSTM 隐藏维度 | 256 |
| LSTM 层数 | 1 |
| 混合精度训练 | FP16 (autocast + GradScaler) |
| 总 Epochs | 30 (Phase1: 10 + Phase2: 20) |

## 5. 验证方案

### LOSO 交叉验证 (Leave-One-Subject-Out)

```
Fold 1:  Train={Sub02-Sub20}  Test={Sub01}
Fold 2:  Train={Sub01,Sub03-Sub20}  Test={Sub02}
...
Fold 20: Train={Sub01-Sub19}  Test={Sub20}
```

- 20 折，每折留出 1 个受试者作为测试集
- 确保测试集受试者从未出现在训练集 → 评估跨受试者泛化能力
- 最终指标 = 20 折预测结果聚合后计算

### 数据流向

```
原始图像 (54,385 帧)
    │
    ├─ 按受试者划分 → 20 折 LOSO
    │
    ├─ 每折内: Sweep → 滑动窗口 (5帧, step=1)
    │   ├─ Train: ~39,000 窗口 → 欠采样 → ~1,940 窗口
    │   └─ Test:  ~2,000 窗口 (保持原始分布)
    │
    ├─ 训练: ResNet-18 + LSTM → 预测
    │
    └─ 聚合所有折预测 → 计算 6 项指标
```

## 6. 代码实现

### 6.1 关键文件

| 文件 | 功能 |
|------|------|
| `config.py` | 所有超参数配置 |
| `model.py` | ResNet-18 + LSTM 模型定义 |
| `train.py` | 训练循环、数据加载、指标计算 |
| `utils/dataset.py` | 数据集类、数据增强、类别权重 |
| `utils/checkpoint.py` | 断点续训工具 |
| `main.py` | 主入口 |

### 6.2 运行命令

```bash
# 完整 20 折 LOSO
python main.py

# 快速验证 (1 折)
python main.py --num_folds 1

# 断点续训
python main.py --resume

# 自定义配置
python main.py --config my_config.yaml
```

## 7. 预期结果与分析方向

### 7.1 基线对比

| 指标 | 随机基线 | 目标 |
|------|----------|------|
| Accuracy | 20% | >50% |
| Weighted F1 | 0.20 | >0.45 |
| Macro F1 | 0.20 | >0.30 |
| Cohen's Kappa | 0.00 | >0.25 |
| AUROC | 0.50 | >0.70 |

### 7.2 分析重点

1. **混淆矩阵对角线：** 各类别是否被正确分类
2. **相邻等级混淆：** Label 1-4 之间是否有系统性误判
3. **Per-class Recall：** 疼痛类 (1-4) 召回率是否显著优于随机
4. **Weighted vs Macro F1 差距：** 差距大说明少数类性能差
5. **Cohen's Kappa：** 是否高于 0.2 (超越"有一定一致性"水平)
