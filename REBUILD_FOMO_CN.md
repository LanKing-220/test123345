# 从 `ei-tennis-openmv-v6` 反推并重建 FOMO（Conda版）

## 1. 先说明反推范围

`ei-tennis-openmv-v6` 里有：

- `trained.tflite`（最终模型）
- `labels.txt`（类别）
- `ei_object_detection.py`（OpenMV 推理后处理）

这些文件可以反推出：

- 任务类型是 FOMO 风格目标检测（热力图 + blob 后处理）
- 类别为：`background`, `Tennis`, `Tennis racket`
- 推理阈值是 `min_confidence = 0.5`

但无法 100% 反推出 Edge Impulse Studio 中的所有训练超参数（如增强策略、epoch、学习率计划、具体骨干网络版本）。

因此这里给你两条可落地路径：

- 路径A（最接近原项目）：用 `tennis-export` 重新导入 Edge Impulse 复训。
- 路径B（完全本地）：用本仓库 YOLO 标注训练一个 FOMO-like 网络并导出 TFLite。

## 2. Conda 重建环境

在项目根目录执行：

```powershell
conda env create -f conda-fomo-env.yml
conda activate fomo-rebuild
```

如果之前装过同名环境：

```powershell
conda env remove -n fomo-rebuild
conda env create -f conda-fomo-env.yml
conda activate fomo-rebuild
```

## 3. 先反推现有 TFLite 的输入输出规格

```powershell
python tools/inspect_tflite.py ei-tennis-openmv-v6/trained.tflite
```

你这个模型的实测结果是：

- 输入：`[1, 98, 98, 3]`
- 输出：`[1, 13, 13, 3]`
- 类型：输入输出都是 `int8`

输出里重点看：

- `Inputs -> shape`：输入分辨率（本项目是 `1x98x98x3`）
- `Outputs -> shape`：输出网格和通道（本项目是 `1x13x13x3`）
- `dtype` 与 `quantization`：量化参数

如果你想机器可读：

```powershell
python tools/inspect_tflite.py ei-tennis-openmv-v6/trained.tflite --json
```

## 4. 本地训练 FOMO-like 模型（从 YOLO 标签）

已提供训练脚本：

- 通用训练：`tools/train_fomo_local.py`
- 短训练（fomo 命名）：`tools/fomo_train_short.py`

默认配置：

- 数据配置：`data/yolo_dataset/data.yaml`
- 标签目录：`data/yolo_labels`
- 输入尺寸：`98`
- 输出网格：`13`

短训练验证命令（5 epoch）：

```powershell
python tools/fomo_train_short.py --epochs 5 --batch-size 16 --out-dir outputs/fomo_short_run
```

通用训练命令：

```powershell
python tools/train_fomo_local.py --epochs 80 --batch-size 16
```

短训练产物会输出到：`outputs/fomo_short_run/`

- `fomo.keras`
- `fomo_float32.tflite`
- `fomo_labels.txt`

## 5. 量化导出（int8 TFLite）

```powershell
python tools/fomo_export_int8.py --keras outputs/fomo_short_run/fomo.keras --train-images data/yolo_dataset/images/train --out outputs/fomo_short_run/fomo_int8.tflite
```

## 6. 原模型 vs 新模型 推理对比（后处理对齐 OpenMV）

```powershell
python tools/fomo_compare_infer.py --image data/yolo_dataset/images/val/img_00001.jpg.6iojvibb.ingestion-789f549d87-2584h.jpg --orig-model ei-tennis-openmv-v6/trained.tflite --new-model outputs/fomo_short_run/fomo_int8.tflite --labels outputs/fomo_short_run/fomo_labels.txt --min-conf 0.5
```

对比脚本会按 `ei_object_detection.py` 的方式输出每个类的中心点和分数。

## 7. 与 OpenMV 脚本对齐

你可以把下面文件替换到 `ei-tennis-openmv-v6` 下做对比测试：

- `outputs/fomo_short_run/fomo_int8.tflite` -> `ei-tennis-openmv-v6/trained.tflite`
- `outputs/fomo_short_run/fomo_labels.txt` -> `ei-tennis-openmv-v6/labels.txt`

随机抽 10 张图片预测命令：

```powershell
python tools/fomo_predict_random10.py --image-dir data/yolo_dataset/images/val --model outputs/fomo_short_run/fomo_int8.tflite --labels outputs/fomo_short_run/fomo_labels.txt --count 10 --seed 42 --min-conf 0.5 --save outputs/fomo_short_run/random10_predictions.md
```

再在 OpenMV 端运行 `ei_object_detection.py` 验证效果。

## 8. 如果你要“最大还原”Edge Impulse版本

`data/tennis-export/` 是官方导出包，建议直接重新导入：

```powershell
edge-impulse-uploader --clean --info-file data/tennis-export/info.labels
```

然后在 Edge Impulse Studio 里重新训练 FOMO 并导出 OpenMV 版本，这条路径和原工程最接近。

## 9. 常见问题

1. `tensorflow` 安装失败
- 先确认 Python 是 3.10。
- 重新建环境，避免在旧环境里混装。

2. 标签看起来异常（比如宽高 > 1）
- 脚本已对中心点进行 `clip`，避免越界崩溃。
- 这会导致极端噪声样本被弱化，但可保证训练可跑。

3. 本地模型和原模型效果不完全一致
- 正常现象。因为原训练超参与增强策略不可完全从导出物反推。
- 可继续微调：`--image-size`、`--grid-size`、`--epochs`、`--lr`。
