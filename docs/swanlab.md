# SwanLab 实验记录说明

本项目已经在 `train.py` 中集成了 SwanLab，可直接记录训练过程、验证指标和最终结果。

## 1. 使用前准备

1. 安装依赖：`pip install swanlab`
2. 登录账号：`swanlab login`
3. 直接启动训练：`python main.py --num_folds 1 --loss ce`

训练开始后，SwanLab 会自动创建一个 run，并同步到项目页面。

## 2. 当前已记录的内容

训练脚本会自动记录以下信息：

- Phase 1 的 `train_loss`、`val_loss`、`train_f1`、`val_f1`
- Phase 2 的 `train_loss`、`val_loss`、`train_f1`、`val_f1`、`backbone_lr`
- 每个 fold 的 `weighted_f1` 和测试受试者编号
- 最终的 `weighted_f1`、`macro_f1`、`cohens_kappa`、`auroc_weighted`
- 最终混淆矩阵图

## 3. 常见图表创建方式

在 SwanLab 页面中点击“创建图表”，通常建议创建以下几类图：

- 折线图：查看 loss 和 F1 随 epoch 的变化
- 标量图：查看最终结果对比
- 柱状图：对比不同 fold 的结果
- 图片图表：展示混淆矩阵

## 4. 推荐的图表字段

如果你想看单条训练曲线，可以优先使用这些字段：

- `fold_0/phase1/train_loss`
- `fold_0/phase1/val_loss`
- `fold_0/phase1/train_f1`
- `fold_0/phase1/val_f1`
- `fold_0/phase2/train_loss`
- `fold_0/phase2/val_loss`
- `fold_0/phase2/train_f1`
- `fold_0/phase2/val_f1`

## 5. 说明

如果后续要做更细的对比，可以继续把每个类别的 recall、AUC 拆成独立标量再记录，这样更容易在 SwanLab 里画柱状图。