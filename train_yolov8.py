"""
YOLOv8n 训练脚本
使用官方ultralytics库
"""
import torch
from ultralytics import YOLO
from pathlib import Path
import json
import os
import cv2
import numpy as np

def convert_labels_to_yolo_v8():
    """将标签转换为YOLO v8格式"""

    label_file = Path('data/tennis-export/info.labels')

    with open(label_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("转换数据集格式...")

    # 创建输出目录
    yolo_dir = Path('data/yolo_dataset')
    yolo_dir.mkdir(exist_ok=True)

    (yolo_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (yolo_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # 类别映射
    class_mapping = {
        'Tennis': 0,
        'Tennis racket': 1,
        'Tennis Racket': 1,
        'Racket': 1
    }

    train_count = 0
    val_count = 0

    for file_info in data['files']:
        img_name = file_info.get('name', '')
        boxes = file_info.get('boundingBoxes', [])
        category = file_info.get('category', 'training')

        # 跳过无标签的图片
        if len(boxes) == 0:
            continue

        # 确定train/val split
        if category == 'testing':
            split = 'val'
            val_count += 1
        else:
            split = 'train'
            train_count += 1

        # 查找原始图片
        for src_dir in [
            Path('data/tennis-export/training'),
            Path('data/tennis-export/testing')
        ]:
            for img_file in src_dir.glob(f'{img_name}*'):
                if img_file.suffix in ['.jpg', '.jpeg', '.png']:
                    # 拷贝图片
                    dst_img = yolo_dir / 'images' / split / f'{img_file.stem}.jpg'
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        cv2.imwrite(str(dst_img), img)

                        # 生成标签文件
                        label_lines = []
                        for box in boxes:
                            box_label = box.get('label', '')
                            x = box.get('x', 0)
                            y = box.get('y', 0)
                            width = box.get('width', 0)
                            height = box.get('height', 0)

                            # 类别ID
                            class_id = class_mapping.get(box_label, 0)

                            # 转换为YOLO格式 (中心坐标和宽高，都归一化到0-1)
                            cx_norm = (x + width / 2) / w
                            cy_norm = (y + height / 2) / h
                            w_norm = width / w
                            h_norm = height / h

                            # 确保在范围内
                            cx_norm = max(0, min(1, cx_norm))
                            cy_norm = max(0, min(1, cy_norm))
                            w_norm = max(0.01, min(1, w_norm))
                            h_norm = max(0.01, min(1, h_norm))

                            label_lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

                        # 写入标签
                        lbl_file = yolo_dir / 'labels' / split / f'{img_file.stem}.txt'
                        with open(lbl_file, 'w') as f:
                            f.write('\n'.join(label_lines))
                    break

    print(f"✓ 训练集: {train_count} 张")
    print(f"✓ 验证集: {val_count} 张")

    # 创建data.yaml
    yaml_content = """path: data/yolo_dataset
train: images/train
val: images/val

nc: 2
names: ['Tennis', 'Racket']
"""

    with open(yolo_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print("✓ data.yaml created")

    return str(yolo_dir / 'data.yaml')

def train_yolov8():
    """训练YOLOv8n模型"""

    # 转换数据集
    yaml_path = convert_labels_to_yolo_v8()

    print("\n" + "="*70)
    print("YOLOv8n Training")
    print("="*70)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=320,
        batch=8,
        patience=5,
        device=0 if torch.cuda.is_available() else 'cpu',
        save=True,
        project='runs/detect',
        name='yolov8n_tennis'
    )

    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {results.save_dir}/weights/best.pt")

    return results

if __name__ == '__main__':
    results = train_yolov8()
