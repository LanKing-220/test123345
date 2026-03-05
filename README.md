# YOLOv10n 轻量级网球检测系统

针对OpenMV优化的轻量级目标检测系统，用于识别网球和网球拍。

## 📁 项目结构

```
毕业设计/
├── yolov10_simple.py      # 轻量级YOLO模型定义（173K参数）
├── train.py               # 训练脚本（10个epoch）
├── predict.py             # 预测和可视化脚本
├── convert_dataset.py     # 数据格式转换工具
├── requirements.txt       # Python依赖包
├── README.md             # 项目说明（本文件）
│
├── data/                 # 数据集目录
│   ├── tennis-export/   # 原始数据
│   └── yolo_labels/     # YOLO格式标签
│
├── models/              # 训练好的模型
│   ├── best_model.pt    # 最佳模型（691KB）
│   └── quantized_model.pt  # 量化模型
│
└── predictions/         # 预测结果
    ├── predictions.txt  # 详细检测结果
    └── *_pred.jpg      # 可视化图像
```

## 🎯 模型特性

- **模型大小**: 173,205 参数（0.66 MB未压缩）
- **输入尺寸**: 320×320×3
- **检测类别**: 2类（网球、网球拍）
  - 网球 → 绿色圆圈标注
  - 网球拍 → 红色矩形标注
- **训练数据**: 168张训练图，45张测试图

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train.py
```

### 3. 运行预测

```bash
python predict.py
```

## 📊 训练结果

- **训练轮数**: 10 epochs
- **最终损失**: 0.0767
- **收敛情况**: 良好（损失从0.72降至0.08）

## 🔍 预测输出

预测结果保存在 `predictions/` 目录：
- `predictions.txt` - 详细检测结果（位置、类别、置信度）
- `*_pred.jpg` - 带标注的可视化图像

## 📝 技术细节

- **框架**: PyTorch 2.0+
- **优化器**: AdamW
- **学习率**: 0.001
- **批大小**: 8
- **损失函数**: BCEWithLogitsLoss
