# SwanLab 实验记录完善计划

## 概述

当前项目已在 `train.py` 中深度集成了 SwanLab，覆盖了超参数记录、训练过程指标、最终评估指标、ECharts 可视化图表和文本摘要。整体框架较为完善，但在以下维度仍有提升空间：

1. **初始化参数**：缺少 `description`、`job_type` 等元信息
2. **训练过程指标**：缺少分类器学习率、每 epoch 训练时间、梯度范数、每类 precision/recall
3. **最终评估可视化**：缺少 ECharts 柱状图、雷达图等对比视图
4. **Config 记录**：部分数据增强和损失函数超参未记录
5. **通知插件**：长时间训练缺少完成/异常通知

---

## 修改文件

- `train.py` — 主要修改文件

---

## 具体改动

### 1. 补全 `swanlab.init` 参数（`_setup_experiment` 函数）

**文件**: [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L195-L230)

在 `swanlab.init()` 调用中增加：

| 参数 | 值 | 用途 |
|------|------|------|
| `description` | 动态生成，包含 backbone、task、loss、fold 数等关键信息 | 实验看板中显示实验目的/上下文 |
| `job_type` | `"train"` | 区分训练/评估/超参搜索等任务类型 |

```python
swanlab.init(
    project="pain-intensity-recognition",
    experiment_name=exp_name,
    group=group_name,
    tags=tags,
    description=(
        f"Pain intensity recognition: {config.pretrained_source} backbone, "
        f"{'binary' if config.binary_mode else f'{config.num_classes}-class'} classification, "
        f"{config.loss_type} loss, {num_folds}-fold LOSO CV"
    ),
    job_type="train",
    config={...},  # 现有 27 个参数 + 下面新增的
)
```

### 2. 补全 Config 超参记录（`_setup_experiment` 函数）

**文件**: [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L200-L229)

在 `config` 字典中新增以下未记录的参数：

```python
# 新增到 config dict
"single_frame": config.single_frame,
"aug_scale": config.aug_scale,
"aug_color_jitter": config.aug_color_jitter,
"aug_rotation_degrees": config.aug_rotation_degrees,
"aug_translate": config.aug_translate,
"lr_scheduler_factor": config.lr_scheduler_factor,
"lr_scheduler_patience": config.lr_scheduler_patience,
```

对于特定损失函数的参数，按需记录：

```python
# 条件新增
if config.loss_type == "focal":
    config_dict["focal_alpha"] = config.focal_alpha
if config.loss_type == "coral":
    config_dict["coral_consistency_weight"] = config.coral_consistency_weight
if config.loss_type == "weighted_ordinal":
    config_dict["ordinal_lambda"] = config.ordinal_lambda
```

### 3. 补全训练过程指标（`train_phase1` 和 `train_phase2`）

**文件**: [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L278-L291), [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L426-L440)

#### 3a. 记录每 epoch 训练时间

在 `train_epoch()` 返回值中增加 elapsed time，或在 phase 函数中自行计时：

```python
import time

# 在每个 epoch 循环开始前
epoch_start = time.time()
# ... train_epoch + evaluate ...
epoch_time = time.time() - epoch_start

log_dict["train/epoch_time"] = epoch_time
```

#### 3b. 记录分类器学习率（Phase 2）

Phase 2 中当前只记录了 `backbone_lr`，补充 `classifier_lr`：

```python
log_dict["val/classifier_lr"] = current_classifier_lr
log_dict[f"fold/{fold_idx}/phase2/classifier_lr"] = current_classifier_lr
```

#### 3c. 记录每类 Precision 和 Recall（每 epoch）

从 `sklearn.metrics` 获取每类的 precision 和 recall：

```python
from sklearn.metrics import precision_score, recall_score

train_precision = precision_score(train_labels, train_preds, average=None, zero_division=0)
train_recall = recall_score(train_labels, train_preds, average=None, zero_division=0)
val_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)
val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)

for i in range(len(val_precision)):
    log_dict[f"val/precision_class_{i}"] = val_precision[i]
    log_dict[f"val/recall_class_{i}"] = val_recall[i]
```

> 注意：这些每 epoch 的 per-class 指标仅在最后一折或指定折中记录，避免指标过多。可通过条件判断 `if fold_idx == 0` 控制。

### 4. 增强最终评估可视化（`_log_final_metrics` 函数）

**文件**: [train.py](file:///home/featurize/work/pain-intensity-recognition/train.py#L679-L728)

#### 4a. ECharts 柱状图 — 各类 Recall 和 AUC 对比

```python
# Per-class recall 柱状图
bar_recall = swanlab.echarts.Bar()
bar_recall.add_xaxis([f"Class {i}" for i in range(config.num_classes)])
bar_recall.add_yaxis("Recall", [round(r, 4) for r in metrics["per_class_recall"]])
swanlab.log({"final/per_class_recall_bar": bar_recall})

# Per-class AUC 柱状图
bar_auc = swanlab.echarts.Bar()
bar_auc.add_xaxis([f"Class {i}" for i in range(config.num_classes)])
bar_auc.add_yaxis("AUC", [round(a, 4) for a in metrics["per_class_auc"]])
swanlab.log({"final/per_class_auc_bar": bar_auc})
```

#### 4b. ECharts 雷达图 — 综合指标对比

```python
# 雷达图：展示主要指标的综合表现
radar = swanlab.echarts.Radar()
radar.add_schema(
    schema=[
        {"name": "Weighted F1", "max": 1.0},
        {"name": "Macro F1", "max": 1.0},
        {"name": "Cohen's Kappa", "max": 1.0},
        {"name": "AUROC", "max": 1.0},
    ]
)
radar.add("Results", [[
    round(metrics["weighted_f1"], 4),
    round(metrics["macro_f1"], 4),
    round(metrics["cohens_kappa"], 4),
    round(metrics["auroc_weighted"], 4),
]])
swanlab.log({"final/radar": radar})
```

#### 4c. 多分类 ROC 曲线

当前仅在二分类模式下记录 ROC/PR 曲线。对于多分类，可以为每个类绘制 one-vs-rest ROC 曲线：

```python
if not config.binary_mode and all_probs.shape[1] > 2:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    classes = range(config.num_classes)
    y_bin = label_binarize(all_labels, classes=list(classes))

    for i in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        # 用 swanlab.echarts 记录多分类 ROC
    # 或者用 matplotlib 绘制多类 ROC 一张图后以 swanlab.Image 记录
```

### 5. 训练汇总表（`_log_final_metrics` 函数末尾）

在 Text Summary 后增加一个 ECharts 表格，汇总所有折的 F1：

```python
# 各折 F1 汇总表
table = swanlab.echarts.Table()
headers = ["Fold", "Test Subject", "Weighted F1"]
rows = []
for i, fold_name in enumerate(completed_folds):
    # 从 fold_results 中提取
    rows.append([i + 1, fold_name, f"{fold_f1:.4f}"])
table.add(headers, rows)
swanlab.log({"final/fold_summary_table": table})
```

> 实现需要在 `run_fold()` 中返回折级 F1 并在主循环中收集。

### 6. 训练环境信息记录（`_setup_experiment` 函数）

在 `swanlab.init` 之后，记录训练环境的静态信息：

```python
import torch
import platform

env_info = {
    "env/pytorch_version": torch.__version__,
    "env/python_version": platform.python_version(),
    "env/gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "env/gpu_vram_gb": round(detect_gpu_vram(), 1) if torch.cuda.is_available() else 0,
    "env/os": platform.system(),
}
swanlab.log(env_info)
```

> 注：也可直接将这些加入 `config` 字典中。

---

## 不改动的部分（已足够完善）

以下方面当前实现已经很好，无需改动：

- **实验命名与分组策略**：experiment_name / group / tags 的设计合理
- **聚合训练曲线**（train/loss, val/loss, train/f1, val/f1）+ 折级详情
- **phase 标识**（train/phase = 1 or 2）
- **最终标量指标**（Weighted F1, Macro F1, Kappa, AUROC, per-class recall/AUC）
- **ECharts 混淆矩阵** 和 **matplotlib 混淆矩阵热力图**
- **Text Summary**
- **ROC/PR 曲线**（二分类模式）
- **`swanlab.finish()` 调用**

---

## 通知插件（可选，用户决定是否启用）

如果用户有需要，可以添加训练完成/异常通知。这是一个**可选**增强，需要用户提供邮箱或飞书配置：

```python
# 可选：通过 CLI 参数 --notify_email 控制
if hasattr(config, 'notify_email') and config.notify_email:
    from swanlab.plugin.notification import EmailCallback
    email_callback = EmailCallback(
        sender_email="...",
        receiver_email=config.notify_email,
        password="...",
        smtp_server="smtp.qq.com",
        port=587,
    )
    swanlab.init(callbacks=[email_callback])
```

> 这需要额外配置 SMTP，暂不纳入必选改动。

---

## 改动优先级

| 优先级 | 改动 | 影响 |
|--------|------|------|
| P0 | 1. 补全 init 参数（description, job_type） | 零成本提升实验管理体验 |
| P0 | 2. 补全 Config 超参记录 | 确保实验可完全复现 |
| P1 | 3a. 每 epoch 训练时间 | 发现性能瓶颈 |
| P1 | 3b. 分类器学习率 | 完整的学习率监控 |
| P1 | 4a. ECharts 柱状图 | 更直观的 per-class 对比 |
| P1 | 4b. ECharts 雷达图 | 综合指标一目了然 |
| P2 | 3c. 每类 precision/recall per epoch | 详细训练监控（可能指标过多） |
| P2 | 4c. 多分类 ROC | 多分类场景增强 |
| P2 | 5. 折汇总表 | 方便查看各折结果 |
| P2 | 6. 环境信息 | 实验复现参考 |
| P3 | 通知插件 | 需用户配置，可选 |

---

## 验证步骤

1. 运行 `python main.py --num_folds 1 --loss ce --binary` 快速验证
2. 在 SwanLab Dashboard 检查：
   - `description` 字段是否正确显示
   - 新增 config 参数是否可见
   - `train/epoch_time` 是否有值
   - ECharts 柱状图和雷达图是否渲染正确
   - 训练环境信息是否记录
3. 确认无 Python 报错或 SwanLab log 异常
