# Pain Intensity Recognition — ResNet + LSTM

基于深度学习的疼痛强度识别模型，支持多种预训练骨干网络 + LSTM 时序建模，
在 MIntPAIN 数据集上进行 5 级疼痛分类。

---

## 模型架构

### Backbone + LSTM

端到端训练：

```
输入: (B, T, C, H, W) — T帧图像序列
  ↓ reshape to (B*T, C, H, W)
Backbone (预训练) → (B*T, feature_dim)
  ↓ reshape to (B, T, feature_dim)
LSTM → (B, hidden_dim)
  ↓ 注意力池化 (可选) / 最后时间步
Classifier → (B, num_classes) 或 (B, K-1) 序数输出
```

### 骨干网络选项

| 选项 | 骨干 | 预训练数据 | 特征维度 | 说明 |
|------|------|-----------|----------|------|
| `imagenet` | ResNet-18 | ImageNet | 512 | **默认**，轻量快速 |
| `vggface2` | InceptionResnetV1 | VGGFace2 | 512 | 人脸识别预训练，需 `facenet-pytorch` |
| `arcface` | InsightFace R50 | MS1MV2 | 512 | ArcFace 人脸识别，需 `onnx2torch` + ONNX 权重 |
| `affectnet` | ResNet-50 | AffectNet | 2048 | 面部表情识别预训练，需下载 `.pt` 权重 |

### 时序窗口采样

每个 sweep 均匀取 K 个窗口（默认 K=3），覆盖疼痛弧线的不同阶段：

```
疼痛时间线: [无痛]...[痛觉建立]...[表情峰值]...[消退]

K=4 窗口分布:
├──w0──┤  ← 刺激开始阶段
    ├──w1──┤  ← 表情建立阶段
        ├──w2──┤  ← 疼痛峰值
            ├──w3──┤  ← 消退/残留
```

每个 sweep 贡献相同数量样本，避免长 sweep 主导训练。

---

## 训练策略

### 两阶段训练

| 阶段 | Backbone | 分类器+LSTM | Epochs | LR |
|------|----------|-------------|--------|----|
| Phase 1 | 冻结 | 训练 | 10 | 1e-3 |
| Phase 2 | 微调 (5e-5) | 训练 (5e-4) | 20 | 分组学习率 + warmup |

**Phase 2 Warmup**: 前 3 个 epoch 线性提升 backbone LR 从 0 → 5e-5，稳定微调初期。

- **Phase 2 起始点**: 从 Phase 1 最佳模型（按 val F1）开始，而非最后一个 epoch
- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau (mode="max", monitor val F1)
- **早停**: patience=7，monitor val F1
- **混合精度**: AMP (GradScaler) + 梯度裁剪 (max_norm=1.0)
- **数据增强**: RandomResizedCrop + RandomHorizontalFlip + ColorJitter + RandomAffine

### 验证策略

- **LOSO 交叉验证**: 留一受试者交叉验证
- **内部验证集**: 从训练受试者中按受试者划分 15% 作为验证集（避免数据泄露）
  - 验证集用于：早停、学习率调度、最佳模型选择
  - 测试集（留出受试者）仅用于最终评估
- **Phase 1/2 最佳模型选择**: Phase 2 必须超越 Phase 1 的 val F1，否则自动回退使用 Phase 1 最佳模型

### 类别不平衡

- **欠采样**（默认 `undersample=True`）：训练集各类别采样至相同数量
- **类别加权**（`class_weight`）：与欠采样互斥，可选 `inverse` / `sqrt_inverse` / `none`

---

## 损失函数

| loss_type | 输出维度 | 序数感知 | 不平衡处理 | 说明 |
|-----------|----------|----------|-----------|------|
| `ce` | K | 否 | class_weight | 标准交叉熵（默认） |
| `corn` | K-1 | 是 | task_weight | 条件序数回归，K-1 个"是否 > k"二分类 |
| `coral` | K-1 | 是（排名一致性） | task_weight | CORN + 一致性惩罚，确保 σ(z_k) ≥ σ(z_{k+1}) |
| `focal` | K | 否 | alpha + gamma | 聚焦困难样本 |
| `weighted_ordinal` | K | 是（距离惩罚） | class_weight + λ | CE + 类别权重 + 序数距离惩罚 |

**序数感知损失**（corn / coral / weighted_ordinal）的核心优势：将 Label 3 误判为 4 的惩罚 < 误判为 0 的惩罚，尊重疼痛等级的自然顺序。

---

## 评估指标

| 指标 | 说明 |
|------|------|
| Weighted F1-score | 对类别不平衡敏感，按样本数加权 |
| Macro F1-score | 各类别同等权重 |
| Per-class Recall | 每个疼痛等级的召回率 |
| Confusion Matrix | 分析相邻等级的混淆情况 |
| Cohen's Kappa | 校正偶然一致性 |
| Multi-class AUROC | 整体区分能力 (OvR) |

---

## 项目结构

```
├── README.md               # 使用说明
├── requirements.txt        # 依赖清单
├── main.py                 # 主入口，命令行解析
├── config.py               # 配置参数（GPU 自动调优）
├── model.py                # 模型定义（4 种骨干 + LSTM + 分类器）
├── train.py                # 训练、评估、指标计算
├── scripts/
│   └── download_models.py  # 下载预训练权重
├── docs/
│   ├── dataset.md          # 数据集说明
│   ├── experiment1.md      # 实验记录
│   ├── experiment2.md      # 实验记录
│   └── swanlab.md          # SwanLab 实验记录说明
└── utils/
    ├── __init__.py
    ├── dataset.py           # 图像序列数据集 / 数据增强
    ├── data_loader.py       # 数据集扫描 / LOSO 分折 / 窗口生成
    ├── checkpoint.py        # 断点续训
    ├── losses.py            # 损失函数（CE / CORN / CORAL / Focal / WeightedOrdinal）
    ├── metrics.py           # 评估指标计算
    ├── schedulers.py        # Warmup + ReduceLROnPlateau
    └── repro.py             # 可复现性工具
```

---

## 环境配置

```bash
pip install -r requirements.txt

# 下载外部预训练权重（可选，仅使用 arcface / affectnet 时需要）
python scripts/download_models.py arcface
python scripts/download_models.py affectnet
```

---

## 使用方法

```bash
# 完整 LOSO 交叉验证（默认：ResNet-18 + CE loss）
python main.py

# 快速验证（1 折）
python main.py --num_folds 1

# 使用不同骨干网络
python main.py --vggface2       # VGGFace2 预训练
python main.py --arcface         # ArcFace 预训练
python main.py --affectnet       # AffectNet 预训练

# 使用不同损失函数（推荐对比实验）
python main.py --loss ce                # 基线：交叉熵
python main.py --loss corn              # 条件序数回归
python main.py --loss coral             # 排名一致序数回归
python main.py --loss focal             # 困难样本聚焦
python main.py --loss weighted_ordinal  # 加权序数 CE

# CORAL / WeightedOrdinal 参数调节
python main.py --loss coral --coral_consistency_weight 0.1
python main.py --loss weighted_ordinal --ordinal_lambda 0.2

# 任务模式
python main.py --binary                 # 二分类（无痛 vs 有痛）
python main.py --single_frame           # 单帧模式（不使用 LSTM）

# 训练控制
python main.py --batch_size 64          # 手动指定 batch size
python main.py --seed 123               # 指定随机种子
python main.py --resume                 # 断点续训
python main.py --config my_config.yaml  # 使用配置文件

# 注意力机制
python main.py --attention              # 使用时序注意力池化
```

### 对比实验矩阵示例

```bash
# 5 种 loss 的对比实验
for loss in ce corn coral focal weighted_ordinal; do
    python main.py --loss $loss --num_folds 20
done

# 4 种 backbone 的对比实验
for bb in vggface2 arcface affectnet; do
    python main.py --$bb --num_folds 20
done
```

---

## 配置参数

### 路径

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mintpain_root` | `/home/featurize/data` | 数据集根目录（可通过 `MINTPAIN_ROOT` 环境变量覆盖） |
| `preprocessed_dir` | `<root>/rgb_preprocessed` | 预处理图像目录 |
| `output_dir` | `<root>/results` | 训练结果输出目录 |
| `pretrained_weights_path` | `./pretrained` | 外部预训练权重存放路径 |

### 模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_classes` | 5 | 疼痛等级数（binary 模式下为 2） |
| `sequence_length` | 5 | 时序窗口帧数 |
| `num_windows_per_sweep` | 3 | 每个 sweep 均匀采样窗口数 (K) |
| `pretrained` | True | 使用预训练权重 |
| `pretrained_source` | `imagenet` | 骨干网络：`imagenet` / `vggface2` / `arcface` / `affectnet` |
| `lstm_hidden_dim` | 256 | LSTM 隐藏维度 |
| `lstm_num_layers` | 1 | LSTM 层数 |
| `dropout` | 0.5 | Dropout 比率 |
| `use_attention_pooling` | False | 使用时序注意力池化 |
| `classifier_hidden_dim` | 0 | 分类器 MLP 隐藏维度（0=单层 Linear） |

### 训练 — Phase 1（分类器训练，backbone 冻结）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase1_epochs` | 10 | Phase 1 训练轮数 |
| `phase1_lr` | 1e-3 | Phase 1 学习率 |

### 训练 — Phase 2（整体微调）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase2_epochs` | 20 | Phase 2 训练轮数 |
| `phase2_backbone_lr` | 5e-5 | Phase 2 backbone 学习率 |
| `phase2_classifier_lr` | 5e-4 | Phase 2 分类器+LSTM 学习率 |
| `warmup_epochs` | 3 | backbone LR warmup 轮数 |

### 训练 — 通用

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 自动 | 批量大小（根据 GPU VRAM 自动调整） |
| `num_workers` | 自动 | DataLoader 进程数（根据 CPU 核心自动调整） |
| `patience` | 7 | 早停耐心值（monitor val F1） |
| `lr_scheduler_factor` | 0.5 | LR 衰减因子 |
| `lr_scheduler_patience` | 2 | LR 衰减耐心值 |
| `seed` | 42 | 随机种子 |
| `deterministic` | True | 是否启用可复现模式（cudnn.deterministic） |

### 数据增强

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `aug_scale` | (0.8, 1.0) | RandomResizedCrop 缩放范围 |
| `aug_color_jitter` | (0.2, 0.2, 0.2) | 亮度/对比度/饱和度变化 |
| `aug_rotation_degrees` | 5 | 随机旋转 ± 度数 |
| `aug_translate` | 0.03 | 随机平移比例 |

### 数据 / 类别不平衡

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `undersample` | True | 训练集欠采样平衡 |
| `class_weight` | `none` | 类别加权：`none` / `inverse` / `sqrt_inverse`（与欠采样互斥） |
| `num_folds` | 0（全部） | LOSO 交叉验证折数 |

### 损失函数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `loss_type` | `ce` | 损失函数：`ce` / `corn` / `coral` / `focal` / `weighted_ordinal` |
| `focal_gamma` | 2.0 | Focal loss gamma（越大越聚焦困难样本） |
| `label_smoothing` | 0.0 | 标签平滑（0=关闭，0.1=中等） |
| `coral_consistency_weight` | 0.05 | CORAL 排名一致性惩罚权重 |
| `ordinal_lambda` | 0.1 | WeightedOrdinalCE 序数距离惩罚权重 |

### 任务模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `binary_mode` | False | 二分类模式（无痛 vs 有痛） |
| `single_frame` | False | 单帧模式（不使用 LSTM） |

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
    ├── fold00_latest.pth      # fold 0 最新 checkpoint
    ├── fold00_epoch001.pth    # fold 0 第 1 个 epoch
    ├── train_progress.json    # 已完成的 folds
    └── ...
```

---

## Checkpoint 断点续训

```bash
# 中断后继续训练
python main.py --resume

# 配合其他参数继续
python main.py --resume --num_folds 10
```
