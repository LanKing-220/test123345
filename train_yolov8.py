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
import shutil


def apply_motion_blur(image, kernel_size=9, angle=0):
    """Apply directional motion blur to simulate camera/object movement."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rotate_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotate_mat, (kernel_size, kernel_size))
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum

    return cv2.filter2D(image, -1, kernel)


def augment_blurry_training_samples(yolo_dir):
    """Create extra blurred training images and duplicate labels for robustness."""
    train_images_dir = Path(yolo_dir) / 'images' / 'train'
    train_labels_dir = Path(yolo_dir) / 'labels' / 'train'

    image_files = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.jpeg')) + list(train_images_dir.glob('*.png'))
    if not image_files:
        print("! No training images found for blur augmentation")
        return 0

    created = 0
    motion_angles = [0, 30, 60, 90, 120, 150]

    for img_file in image_files:
        if '_blur_g' in img_file.stem or '_blur_m' in img_file.stem:
            continue

        label_file = train_labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # Gaussian blur sample
        gaussian_path = img_file.with_name(f"{img_file.stem}_blur_g.jpg")
        gaussian_label = train_labels_dir / f"{img_file.stem}_blur_g.txt"
        if not gaussian_path.exists():
            blur_g = cv2.GaussianBlur(img, (5, 5), 1.2)
            cv2.imwrite(str(gaussian_path), blur_g)
            shutil.copy2(label_file, gaussian_label)
            created += 1

        # Motion blur sample
        motion_path = img_file.with_name(f"{img_file.stem}_blur_m.jpg")
        motion_label = train_labels_dir / f"{img_file.stem}_blur_m.txt"
        if not motion_path.exists():
            angle = motion_angles[created % len(motion_angles)]
            blur_m = apply_motion_blur(img, kernel_size=9, angle=angle)
            cv2.imwrite(str(motion_path), blur_m)
            shutil.copy2(label_file, motion_label)
            created += 1

    print(f"✓ Blurry augmentation created: {created} images")
    return created

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
    yolo_root = Path(yaml_path).parent
    augment_blurry_training_samples(yolo_root)

    print("\n" + "="*70)
    print("YOLOv8n Training")
    print("="*70)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 训练
    results = model.train(
        data=yaml_path,
        epochs=10,
        imgsz=640,
        batch=8,
        patience=20,
        device=0 if torch.cuda.is_available() else 'cpu',
        optimizer='AdamW',
        lr0=0.003,
        lrf=0.1,
        warmup_epochs=3,
        weight_decay=0.0005,
        cos_lr=True,
        mosaic=0.8,
        mixup=0.15,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        close_mosaic=10,
        erasing=0.2,
        label_smoothing=0.05,
        seed=42,
        save=True,
        project='runs/detect',
        name='yolov8n_tennis'
    )

    print("\n✓ Training completed!")
    print(f"✓ Best model saved to: {results.save_dir}/weights/best.pt")

    return results

if __name__ == '__main__':
    results = train_yolov8()
