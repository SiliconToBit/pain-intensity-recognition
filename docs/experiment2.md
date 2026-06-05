# 实验二：ArcFace R50 + LSTM 疼痛强度识别（二分类）

## 1. 实验目标

在实验一（ResNet-18 ImageNet）的基础上，将特征提取器替换为 InsightFace ArcFace R50 预训练模型，评估人脸识别预训练特征对疼痛识别任务的效果。

**任务**：二分类（有痛 vs 无痛）
**评估方式**：5 折 LOSO（Leave-One-Subject-Out）交叉验证

## 2. 模型架构

### 2.1 整体结构

```
输入: (B, T, C, H, W) — 批次 × T帧 × 3通道 × 224 × 224
  │
  ├─ reshape → (B×T, 3, 224, 224)
  │
  ├─ ArcFace R50 Feature Extractor
  │   ├─ 预处理: ImageNet归一化 → [0,1] → RGB→BGR → [-1,1]
  │   ├─ Resize → 112×112
  │   ├─ InsightFace R50 Backbone (via onnx2torch)
  │   │   ├─ Conv_0 + PReLU + BN
  │   │   ├─ 24× Residual Blocks (类似 ResNet-50 Bottleneck)
  │   │   ├─ Gemm (FC: 25088→512)
  │   │   └─ BatchNorm1d(512)
  │   └─ 输出: 512-dim embedding
  │
  ├─ reshape → (B, T, 512)
  │
  ├─ LSTM (input=512, hidden=256, layers=1)
  │   └─ 取最后时步输出 → (B, 256)
  │
  └─ Classifier
      ├─ Dropout(0.5)
      └─ Linear(256 → 2) → logits
```

### 2.2 ArcFace R50 详情

| 属性 | 值 |
|------|-----|
| 预训练数据 | MS1MV2（5.8M 图像，85K 身份） |
| 损失函数 | Additive Angular Margin (ArcFace) |
| 输入尺寸 | 112 × 112 BGR |
| 输出维度 | 512 |
| 总参数量 | ~25M |
| 模型来源 | insightface `buffalo_l` 包 |
| 加载方式 | `onnx2torch.convert()` 直接加载 ONNX |

### 2.3 预处理链

数据集 pipeline 输出 ImageNet 归一化的 RGB 图像，ArcFace 需要 BGR [-1,1] 输入：

```
ImageNet归一化图像 (RGB, mean/std归一化)
  → ×std + mean 反归一化 → [0, 1]
  → interpolate 224→112
  → RGB→BGR 通道翻转
  → ×2 - 1 → [-1, 1]
```

### 2.4 训练策略

两阶段训练：

| 阶段 | Backbone | 分类器 | LR | Epochs |
|------|----------|--------|-----|--------|
| Phase 1 | 冻结 | 训练 | 1e-3 | 10 (early stop) |
| Phase 2 | 微调 (warmup) | 训练 | backbone: 5e-5→1.25e-5, classifier: 5e-4 | 30 (early stop) |

- Phase 2 batch size 从 96 降至 **32**（避免 OOM）
- 学习率调度：ReduceLROnPlateau (factor=0.5, patience=2)
- Warmup：backbone LR 从 0 线性升至 5e-5（前 3 个 epoch）

## 3. 数据集

- **MIntPAIN** 数据集，20 个受试者，3122 个 sweep
- 二分类重映射：Label 0 → 无痛(0)，Label 1-4 → 有痛(1)
- 欠采样多数类使类别平衡
- 滑动窗口：5 帧，步长 2

## 4. 实验结果

### 4.1 五折 LOSO 汇总

| 指标 | ArcFace R50 | ImageNet R18 | 提升 |
|------|:-----------:|:------------:|:----:|
| Weighted F1 | **0.527** | 0.430 | +0.097 |
| Macro F1 | **0.527** | 0.430 | +0.097 |
| Cohen's Kappa | **0.089** | -0.002 | +0.091 |
| AUROC (weighted) | **0.572** | 0.510 | +0.062 |

### 4.2 混淆矩阵（5折汇总，3752 样本）

```
              预测无痛   预测有痛
真实无痛 (1876)   1385       491
真实有痛 (1876)   1218       658
```

### 4.3 各折详细结果

| Fold | Test Subject | Weighted F1 | Phase 1 Best Epoch | Phase 2 Best Epoch |
|------|-------------|:-----------:|:------------------:|:------------------:|
| 1 | Sub01 | 0.4658 | 9 | 13 |
| 2 | Sub02 | 0.4982 | 9 | 13 |
| 3 | Sub03 | 0.5089 | 9 | 13 |
| 4 | Sub04 | 0.4675 | 8 | 12 |
| 5 | Sub05 | **0.5690** | 9 | 12 |
| **平均** | | **0.502** | | |

### 4.4 Per-class 分析

| 类别 | Precision | Recall | F1 | Support |
|------|:---------:|:------:|:--:|:-------:|
| 无痛 (0) | 0.532 | 0.738 | 0.618 | 1876 |
| 有痛 (1) | 0.573 | 0.351 | 0.435 | 1876 |

## 5. 技术问题与解决方案

### 5.1 权重不兼容（已解决）

**问题**：InsightFace R50 架构（[3,4,14,3] blocks）与 torchvision ResNet-50（[3,4,6,3]）完全不同，无法直接加载权重。

**解决方案**：使用 `onnx2torch` 直接加载完整的 ONNX 模型，保留完整计算图支持微调。

### 5.2 buffalo_s 缺少 R50 模型（已解决）

**问题**：`buffalo_s.zip` 只包含 `w600k_mbf.onnx`（MobileFaceNet），不含 `w600k_r50.onnx`。

**解决方案**：改用 `buffalo_l.zip`，包含完整的 `w600k_r50.onnx`（166MB）。

### 5.3 Phase 2 OOM（已解决）

**问题**：ArcFace R50 模型参数量大（~25M），解冻后 batch_size=96 导致显存不足（RTX 3080 9.6GB）。

**解决方案**：
- Phase 2 自动将 batch_size 从 96 降至 32
- 启用 gradient checkpointing

### 5.4 Checkpoint 兼容性（已解决）

**问题**：Resume 时旧 checkpoint（ResNet-18 架构）与新模型（onnx2torch 架构）不兼容。

**解决方案**：`train.py` 中添加 try-except，遇到不兼容 checkpoint 自动跳过，从头训练。

## 6. 结论

1. **ArcFace 优于 ImageNet**：F1 从 0.43 提升到 0.53（+23%），说明人脸预训练特征对疼痛识别有一定帮助
2. **仍有提升空间**：Kappa=0.089 表明模型接近随机水平，身份特征中的疼痛信号较弱
3. **偏向无痛预测**：无痛 recall 74% vs 有痛 recall 35%，模型倾向于保守预测
4. **onnx2torch 方案可行**：成功将 insightface ONNX 模型转为可微调的 PyTorch Module，160/160 参数均可训练

## 7. 后续优化方向

- [ ] 调整数据增强策略（减少 RandomAffine 对面部结构的破坏）
- [ ] 全量 20-fold LOSO 交叉验证
- [ ] 调整分类阈值以平衡两类 recall
- [ ] 尝试冻结更多层，只微调后几层
- [ ] 对比 VGGFace2 预训练（FaceNet）的效果
- [ ] 尝试更大的 LSTM hidden dim 或多层 LSTM
