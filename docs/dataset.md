# MIntPAIN 数据集说明

## 1. 数据集概述

MIntPAIN (Multimodal Intensity Pain) 是一个多模态疼痛强度数据集，本项目使用其中的 **RGB 面部图像** 部分，用于 5 级疼痛强度分类任务。

| 属性 | 值 |
|------|-----|
| 受试者数量 | 20 人 |
| 总帧数 | 54,385 |
| 总 sweeps | 3,122 |
| 每位受试者 sweeps | 146 - 160 |
| 图像尺寸 | 224 × 224 RGB |
| 图像格式 | JPEG (quality=95) |
| 疼痛等级 | 0-4 (5 类) |

## 2. 目录结构

```
mintpain/
├── rgb_preprocessed/                    # 预处理后的图像
│   ├── preprocessing_stats.json         # 预处理统计信息
│   ├── Sub1 Daniel Simonsen/
│   │   ├── Annotated_data_Sub01_Trial01/
│   │   │   ├── Sub01_Trial01_Sweep01_Label0/
│   │   │   │   └── rgb/
│   │   │   │       ├── RGB-11-51-02-0121.jpg
│   │   │   │       ├── RGB-11-51-02-0782.jpg
│   │   │   │       └── ...              (8-28 帧/sweep)
│   │   │   ├── Sub01_Trial01_Sweep01_Label3/
│   │   │   ├── Sub01_Trial01_Sweep02_Label0/
│   │   │   └── ...                      (80 sweeps/trial)
│   │   └── Annotated_data_Sub01_Trial02/
│   └── Sub2 ... Sub20/
```

**命名规则：**
- 受试者目录：`Sub{N} {姓名}`
- Trial 目录：`Annotated_data_Sub{NN}_Trial{NN}`
- Sweep 目录：`Sub{NN}_Trial{NN}_Sweep{NN}_Label{L}`
  - `Label{L}` 表示疼痛等级 (0-4)
- 帧文件：`RGB-{HH}-{MM}-{SS}-{ms}.jpg` (时间戳格式)

## 3. 类别分布

| 疼痛等级 | 含义 | Sweeps 数 | 帧数 | 占比 |
|----------|------|-----------|------|------|
| Label 0 | 无痛 | 1,561 | 27,207 | 50.0% |
| Label 1 | 轻微疼痛 | 388 | 6,767 | 12.4% |
| Label 2 | 中度疼痛 | 392 | 6,742 | 12.4% |
| Label 3 | 较强疼痛 | 388 | 6,825 | 12.5% |
| Label 4 | 剧烈疼痛 | 393 | 6,844 | 12.6% |

**不平衡比：** Class 0 数量约为其他类的 **4 倍**。

```
类别分布柱状图 (sweep 级别):

Label 0  ████████████████████████████████████████████████  1561
Label 1  ████████████                                       388
Label 2  ████████████                                       392
Label 3  ████████████                                       388
Label 4  ████████████                                       393
```

## 4. 受试者信息

| 编号 | 姓名 | Sweeps | 帧数 |
|------|------|--------|------|
| Sub01 | Daniel Simonsen | 160 | 2,902 |
| Sub02 | Fabrio Jure | 156 | 2,778 |
| Sub03 | Federico Arguissan | 158 | 2,976 |
| Sub04 | Morten Hogh | 160 | 3,122 |
| Sub05 | Enrico de Martino | 160 | 3,034 |
| Sub06 | Davide Ligai | 152 | 2,620 |
| Sub07 | Hans Jorgen Kabs | 154 | 2,620 |
| Sub08 | Dennis Boye Larsen | 152 | 2,576 |
| Sub09 | Mads Jocumsen | 156 | 2,546 |
| Sub10 | Alessandro Ranieri | 160 | 2,738 |
| Sub11 | Rasmus Elbaek | 158 | 2,615 |
| Sub12 | Mikkel Thogarien | 160 | 2,698 |
| Sub13 | Gazi Arifur Rahman | 156 | 2,592 |
| Sub14 | Ramin Irani | 160 | 2,700 |
| Sub15 | Mohammad Ahsanul Haque | 146 | 2,538 |
| Sub16 | Soren Heth | 158 | 2,712 |
| Sub17 | Rasmus K Nielsen | 154 | 2,602 |
| Sub18 | Neils Krogh Madsen | 158 | 2,692 |
| Sub19 | Mohammad Reza | 150 | 2,720 |
| Sub20 | Simon Manget | 154 | 2,604 |

## 5. 预处理流程

图像已经过以下预处理步骤：

| 步骤 | 方法 | 说明 |
|------|------|------|
| 人脸检测 | YuNet (ONNX) | 置信度阈值 0.6, NMS 阈值 0.3 |
| 备选检测 | Haar Cascade | YuNet 失败时的回退方案 |
| 裁剪策略 | 人脸区域裁剪 + 缩放 | 输出 224×224 |
| 时序一致性 | 最大位移 100px, 面积比 3.0 | 防止帧间人脸跳变 |
| 输出格式 | JPEG quality=95 | RGB 三通道 |

**检测统计：**
- YuNet 检测成功：54,385 / 54,385 (100%)
- Haar 回退：0
- 全图缩放回退：0
- 损坏图像：0

## 6. 数据特点与挑战

1. **严重类别不平衡：** 无痛类占 50%，疼痛类各仅 12.5%
2. **个体差异大：** 20 名受试者来自不同种族/性别，面部表情差异显著
3. **相邻等级混淆：** 疼痛等级 1-4 之间界限模糊，人类标注者也难以区分
4. **时序连续性：** 同一 sweep 内的帧具有时序关系，可利用滑动窗口建模
5. **帧数不均匀：** 不同 sweep 的帧数从 8 到 28 不等
