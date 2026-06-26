# AffectNet R50 单帧二分类评估方案

## 问题

运行 `python main.py --affectnet --single_frame` 时 Phase 2 被系统 kill。

**现象**：
- Phase 1 batch_size=542，冻结 backbone，8 个 epoch 正常完成
- Phase 2 batch_size=271（减半），解冻 backbone，评估刚开始即被 kill
- GPU: RTX 2080 Ti，22.5GB VRAM

**根因分析**：
1. `single_frame` 在 `Config.__init__()` 完成后才设置，auto-tune 按 LSTM 模式估算（per_sample_gb=0.17），对单帧模式偏保守但实际 batch_size=542 异常大（代码 cap 在 128，可能是运行时代码与磁盘不一致）
2. Phase 2 解冻 ResNet-50（23.5M 参数），Adam 优化器需额外 ~400MB（状态 + 梯度），仅减半不够
3. Phase 2 评估时虽是 no_grad，但模型 + 优化器状态已占用大量显存，batch_size=271 的中间激活仍可能触发 OOM 或系统级 kill

## 修改计划

### 修改 1：config.py — auto-tune 感知 single_frame 模式

**文件**: [config.py](file:///home/featurize/work/pain-intensity-recognition/config.py)

`Config.__init__` 新增 `single_frame` 参数，在 `_auto_tune()` 前设置。

单帧模式每样本只处理 1 张图（vs LSTM 模式 5 帧 + LSTM forward），per-sample VRAM 更低：
- affectnet/arcface 单帧：`per_sample_gb = 0.04`（vs LSTM 的 0.17）
- imagenet/vggface2 单帧：`per_sample_gb = 0.015`（vs LSTM 的 0.06）

22.5GB 预期结果：
- Phase 1：`min(128, 21.5/0.04)` = 128
- Phase 2：128 // 4 = 32（见修改 3）

### 修改 2：main.py — 提前传 single_frame 给 Config

**文件**: [main.py](file:///home/featurize/work/pain-intensity-recognition/main.py)

```python
config = Config(
    args.config,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pretrained_source=...,
    single_frame=args.single_frame,  # 新增
)
```

### 修改 3：train.py — Phase 2 缩减更激进

**文件**: [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L375-L381)

当前 arcface/affectnet Phase 2 只 `// 2`，改为 `// 4`：

```python
# 修改前
phase2_batch_size = max(4, phase1_batch_size // 2)

# 修改后
phase2_batch_size = max(4, phase1_batch_size // 4)
```

## 预期 batch_size（22.5GB VRAM）

| 阶段 | 修改前 | 修改后 |
|------|--------|--------|
| Phase 1 (冻结) | 542 | 128 |
| Phase 2 (解冻) | 271 (kill) | 32 |

## 修改后运行命令

```bash
# 5 折快速验证
python main.py --affectnet --binary --single_frame --num_folds 5

# 全量 20 折 LOSO
python main.py --affectnet --binary --single_frame
```

## 验证步骤

1. 运行修改后代码，确认 Phase 1/2 均不 OOM
2. 检查日志中 batch_size 输出是否为 128/32
3. 完成 5 折 LOSO，对比 ArcFace R50 LSTM 二分类（F1=0.527）
