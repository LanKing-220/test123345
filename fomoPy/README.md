# fomoPy

`fomoPy` is a local FOMO-like training workspace for the Edge Impulse export at `../data/tennis-export`.

Important:

- Edge Impulse's official FOMO training pipeline is not published as a standalone local training repository.
- This workspace uses the official Edge Impulse exported dataset format and converts it into a local YOLO-style label layout for training.
- The resulting model is a local FOMO-like TensorFlow/TFLite model, not a bit-for-bit reproduction of Edge Impulse Studio training.

## 1. Create the environment

```powershell
conda env create -f environment.yml
conda activate fomo-rebuild
```

If the environment already exists:

```powershell
conda env update -f environment.yml --prune
conda activate fomo-rebuild
```

## 2. Convert the exported dataset

Run from the repository root:

```powershell
python fomoPy/prepare_tennis_export.py
```

This creates:

- `fomoPy/data/yolo_labels/training`
- `fomoPy/data/yolo_labels/testing`
- `fomoPy/data/yolo_dataset/data.yaml`

The images stay in the original exported dataset under `data/tennis-export`.

## 3. Quick training smoke test

```powershell
python fomoPy/train_quick.py --epochs 5 --batch-size 8
```

## 4. Full training

```powershell
python fomoPy/train.py --epochs 80 --batch-size 16 --image-size 96 --grid-size 12
```

Recommended for OpenMV alignment:

- `--image-size 96 --grid-size 12`
- or inspect your original exported model and align the input/output shapes manually

## 5. Outputs

Training outputs are written to:

- `fomoPy/outputs/fomo_local/fomo_like.keras`
- `fomoPy/outputs/fomo_local/fomo_like_float32.tflite`
- `fomoPy/outputs/fomo_local/labels.txt`

## 6. Dataset labels

Detected classes are read directly from the Edge Impulse bounding-box export. For your tennis dataset, the classes currently include:

- `Tennis`
- `Tennis player`
- `Tennis racket`
