# fomoPy 使用说明

`fomoPy` 现在只保留一套正式训练流程，目标是：

- 读取 `data/tennis-export` 这份 Edge Impulse 导出数据
- 训练一个本地 FOMO-like 模型
- 同时导出两份模型：
  - OpenMV 4.8.1 可直接测试的 `float32 tflite`
  - 适合 OpenMV 部署的 `int8 tflite`

不再保留：

- `train_quick.py`
- 多种 `model-size` 版本切换
- 单独的 `export_int8.py`

因为 `train.py` 训练完成后已经会直接导出 OpenMV 对应的 `int8` 版本。

## 1. 创建环境

在项目根目录执行：

```powershell
conda env create -f fomoPy/environment.yml
conda activate fomo-rebuild
```

## 2. 准备数据

执行：

```powershell
python fomoPy/prepare_tennis_export.py
```

它会把 `data/tennis-export` 转成训练脚本可直接读取的标签和配置文件。

生成内容：

- `fomoPy/data/yolo_labels/training`
- `fomoPy/data/yolo_labels/testing`
- `fomoPy/data/yolo_dataset/data.yaml`

当前类别：

- `Tennis`
- `Tennis player`
- `Tennis racket`

## 3. 正式训练

执行：

```powershell
python fomoPy/train.py --epochs 80 --batch-size 16 --image-size 96 --grid-size 12 --metric-threshold 0.35 --label-mode soft-box --bbox-radius-scale 0.25 --auto-class-weight --focus-classes "Tennis player,Tennis racket" --focus-multiplier 3 --out-dir fomoPy/outputs/fomo_local
```

这版 `train.py` 会根据 `image-size / grid-size` 自动对齐输出网格。
推荐继续从 `96 / 12` 开始，兼顾 OpenMV 体积与召回率。

## 4. 输出文件

训练完成后，会在输出目录中生成：

- `fomo_like.keras`
- `fomo_like_float32.tflite`
- `fomo_like_int8.tflite`
- `labels.txt`
- `fomo_like.h5`（仅在显式加 `--save-h5` 时输出）

说明：

- `fomo_like.keras`
  训练保存文件，用于继续训练或调试。

- `fomo_like.h5`
  兼容旧工具链时可选输出，默认不保存，避免 Windows 路径下卡住训练流程。

- `fomo_like_float32.tflite`
  OpenMV 4.8.1 的 float 推理可直接测试，也适合本地调试和分析。

- `fomo_like_int8.tflite`
  OpenMV 对应版本，优先用于部署。

- `labels.txt`
  和模型一起使用的类别文件。

## 5. OpenMV 使用

如果你要替换 OpenMV 工程中的模型，优先使用：

- `fomoPy/outputs/fomo_local/fomo_like_int8.tflite`
- `fomoPy/outputs/fomo_local/labels.txt`

如果你的 OpenMV 固件已经升级到 `4.8.1`，也可以先把：

- `fomoPy/outputs/fomo_local/fomo_like_float32.tflite`

复制成设备上的 `trained.tflite` 直接做 float 版本调试。

不要优先直接部署：

- `fomo_like.keras`
- `fomo_like_float32.tflite` 到长期量产版本

## 6. 评估训练集和测试集

执行：

```powershell
python fomoPy/eval_testset.py --model fomoPy/outputs/fomo_local/fomo_like_int8.tflite --threshold 0.35 --class-thresholds "0.35,0.40,0.35" --label-mode soft-box --report fomoPy/outputs/fomo_local/eval_metrics.json
```

这个脚本会同时输出：

- 训练集结果
- 测试集结果

并保存到：

- `fomoPy/outputs/fomo_local/eval_metrics.json`

## 7. 推荐最简流程

```powershell
conda activate fomo-rebuild
python fomoPy/prepare_tennis_export.py
python fomoPy/train.py --epochs 80 --batch-size 16 --image-size 96 --grid-size 12 --metric-threshold 0.35 --label-mode soft-box --bbox-radius-scale 0.25 --auto-class-weight --focus-classes "Tennis player,Tennis racket" --focus-multiplier 3 --out-dir fomoPy/outputs/fomo_local
python fomoPy/eval_testset.py --model fomoPy/outputs/fomo_local/fomo_like_int8.tflite --threshold 0.35 --class-thresholds "0.35,0.40,0.35" --label-mode soft-box --report fomoPy/outputs/fomo_local/eval_metrics.json
```

## 8. 当前定位

这套代码是“本地可训练、可导出 OpenMV 对应 int8 模型”的简化版本。

它不是 Edge Impulse 官方 Studio 后端训练器本体，但已经保留了你现在实际需要的核心链路：

- 数据准备
- 正式训练
- OpenMV 对应导出
- 训练集 / 测试集评估
