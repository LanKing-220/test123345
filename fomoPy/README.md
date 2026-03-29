# fomoPy

`fomoPy` 现在只保留一套最终版本：

- `train.py`：本地 FOMO-like 训练入口
- `eval_testset.py`：本地测试集评估入口

其余 `official` 路线、批量实验脚本、外部数据增强脚本都已移除，避免项目结构继续发散。

## 保留文件

- `train.py`
- `eval_testset.py`
- `local_fomo_common.py`
- `local_train_common.py`
- `prepare_tennis_export.py`
- `environment.yml`

## 数据准备

先把 Edge Impulse 导出的 `data/tennis-export` 转成 YOLO 标签：

```powershell
python fomoPy/prepare_tennis_export.py
```

生成：

- `fomoPy/data/yolo_labels/training`
- `fomoPy/data/yolo_labels/testing`
- `fomoPy/data/yolo_dataset/data.yaml`

类别：

- `Tennis`
- `Tennis player`
- `Tennis racket`

## 训练

推荐命令：

```powershell
python fomoPy/train.py --data-yaml fomoPy/data/yolo_dataset/data.yaml --labels-dir fomoPy/data/yolo_labels --epochs 90 --batch-size 16 --image-size 96 --grid-size 12 --lr 0.0012 --optimizer adamw --weight-decay 0.0002 --aug-multiplier 2 --hflip-prob 0.5 --bg-weight 0.18 --fg-weight 3.0 --focal-gamma 1.75 --metric-threshold 0.28 --label-mode soft-box --bbox-radius-scale 0.35 --min-target-radius 1 --max-target-radius 4 --stage-width-scale 1.25 --head-filters 48 --refine-blocks 1 --dropout-rate 0.05 --dice-weight 0.15 --auto-class-weight --min-class-weight 1.0 --max-class-weight 5.0 --focus-classes "Tennis player,Tennis racket" --focus-multiplier 2 --out-dir fomoPy/outputs/fomo_local
```

训练输出：

- `fomo_like.keras`
- `fomo_like_float32.tflite`
- `fomo_like_int8.tflite`
- `labels.txt`
- `training_history.json`
- `config.json`

## 评估

```powershell
python fomoPy/eval_testset.py --model fomoPy/outputs/fomo_local/fomo_like_int8.tflite --data-yaml fomoPy/data/yolo_dataset/data.yaml --labels-dir fomoPy/data/yolo_labels --threshold 0.30 --class-thresholds "0.30,0.26,0.26" --label-mode soft-box --bbox-radius-scale 0.35 --min-target-radius 1 --max-target-radius 4 --report fomoPy/outputs/fomo_local/eval_metrics.json
```

## OpenMV

优先使用：

- `fomo_like_int8.tflite`

如果是 OpenMV 4.8.1 浮点调试，也可以测试：

- `fomo_like_float32.tflite`
