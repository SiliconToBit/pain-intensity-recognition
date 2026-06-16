# SwanLab 实验记录说明

本项目在 `train.py` 中集成了 SwanLab，自动记录训练过程、验证指标和最终结果。

## 1. 使用前准备

1. 安装依赖：`pip install swanlab`
2. 登录账号：`swanlab login`
3. 启动训练：`python main.py --num_folds 1 --loss ce`

## 2. 实验组织方式

### 命名规则

```
experiment_name: {backbone}_{task}_{loss}[_{variant}]
```

示例：
- `imagenet_5class_ce` — ImageNet ResNet-18, 五分类, CE 损失
- `arcface_binary_ce` — ArcFace R50, 二分类, CE 损失
- `arcface_binary_corn` — ArcFace R50, 二分类, Corn 序数回归
- `arcface_binary_ce_attention` — 加时序注意力池化

### Group / Tags

| 字段 | 用途 | 示例 |
|------|------|------|
| `group` | 同一实验的多次运行分组对比 | `arcface_binary_ce` |
| `tags` | 多维筛选 | `["arcface", "binary", "ce", "undersample"]` |

在 SwanLab Dashboard 中，可按 tag 筛选、按 group 分组对比。

## 3. 记录的指标

### 聚合训练曲线（跨折对比用）

| Key | 说明 |
|-----|------|
| `train/loss` | 训练损失（全局 epoch step） |
| `train/f1` | 训练加权 F1 |
| `val/loss` | 验证损失 |
| `val/f1` | 验证加权 F1 |
| `val/backbone_lr` | Backbone 学习率（Phase 2） |
| `train/phase` | 当前阶段（1=冻结 / 2=微调） |

### 折级详情（drill-down 用）

| Key | 说明 |
|-----|------|
| `fold/{i}/phase1/train_loss` | 第 i 折 Phase 1 训练损失 |
| `fold/{i}/phase1/val_f1` | 第 i 折 Phase 1 验证 F1 |
| `fold/{i}/phase2/*` | Phase 2 同理 |
| `fold/weighted_f1` | 每折最终 F1（step = fold 编号） |

### 最终结果

| Key | 说明 |
|-----|------|
| `final/weighted_f1` | 聚合加权 F1 |
| `final/macro_f1` | 聚合宏平均 F1 |
| `final/cohens_kappa` | Cohen's Kappa |
| `final/auroc_weighted` | 加权 AUROC |
| `final/recall_class_{i}` | 第 i 类召回率（独立标量，方便柱状图） |
| `final/auc_class_{i}` | 第 i 类 AUC |
| `final/confusion_matrix` | 混淆矩阵热力图 |
| `summary` | Markdown 文本总结 |

## 4. 推荐图表

在 SwanLab Dashboard 中建议创建：

- **折线图**：`val/f1` + `train/f1` — 观察训练趋势
- **折线图**：`val/loss` + `train/loss` — 观察过拟合
- **柱状图**：`final/recall_class_*` — 对比各类召回率
- **柱状图**：`final/auc_class_*` — 对比各类 AUC
- **热力图**：`final/confusion_matrix` — 混淆矩阵
- **表格**：按 group 对比 `final/weighted_f1`

## 5. 实验划分建议

每个唯一的 (backbone × task × loss × variant) 组合 = 一个实验：

```
Project: pain-intensity-recognition
│
├── Group: imagenet_5class_ce
│   ├── Run: 5-fold
│   └── Run: 20-fold
│
├── Group: arcface_binary_ce
│   ├── Run: 5-fold
│   └── Run: 20-fold
│
├── Group: arcface_binary_corn       ← 换 loss
│   └── Run: 5-fold
│
└── Group: arcface_binary_ce_attention  ← 加 attention
    └── Run: 5-fold
```

多折交叉验证是同一实验内的评估方式，不需要拆成独立实验。
