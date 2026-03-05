"""
YOLOv8n 预测脚本 - 英文标注
"""
from ultralytics import YOLO
from pathlib import Path
import cv2

def predict_yolov8():
    """使用YOLOv8进行预测"""

    print("="*70)
    print("YOLOv8n Tennis Detection - Prediction")
    print("="*70)

    # 加载最佳模型
    model_path = 'runs/detect/runs/detect/yolov8n_tennis2/weights/best.pt'

    if not Path(model_path).exists():
        print(f"✗ Model not found: {model_path}")
        return

    model = YOLO(model_path)
    print(f"\n✓ Loaded model: {model_path}")

    # 查找测试图片
    test_dir = Path('data/tennis-export/testing')
    test_images = list(test_dir.glob('*.jpg'))[:10]

    results_dir = Path('predictions_v8')
    results_dir.mkdir(exist_ok=True)

    print(f"✓ Test images: {len(test_images)}\n")

    total_detections = 0
    tennis_count = 0
    racket_count = 0

    # 预测并保存结果
    with open(results_dir / 'predictions.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("YOLOv8n Tennis Detection Results\n")
        f.write("="*70 + "\n\n")

        for idx, img_path in enumerate(test_images, 1):
            # 预测
            results = model.predict(source=str(img_path), conf=0.5, save=False)

            # 读取图片
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # 提取检测结果
            detections = []
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())

                    # 边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    bw = int(x2 - x1)
                    bh = int(y2 - y1)

                    class_names = ['Tennis', 'Racket']
                    detections.append({
                        'class': cls_id,
                        'class_name': class_names[cls_id],
                        'confidence': conf,
                        'x': cx,
                        'y': cy,
                        'w': bw,
                        'h': bh
                    })

            # 统计
            tennis_det = sum(1 for d in detections if d['class'] == 0)
            racket_det = sum(1 for d in detections if d['class'] == 1)

            total_detections += len(detections)
            tennis_count += tennis_det
            racket_count += racket_det

            print(f'[{idx}/{len(test_images)}] {img_path.name}')
            print(f'  Detections: {len(detections)} (Tennis: {tennis_det}, Racket: {racket_det})')

            # 写入文件
            f.write(f"Image {idx}: {img_path.name}\n")
            f.write(f"Size: {w}x{h}\n")
            f.write(f"Detections: {len(detections)}\n")

            if detections:
                f.write("Results:\n")
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                for i, det in enumerate(detections, 1):
                    f.write(f"  [{i}] {det['class_name']}\n")
                    f.write(f"      Center: ({det['x']}, {det['y']})\n")
                    f.write(f"      Size: {det['w']}x{det['h']}\n")
                    f.write(f"      Confidence: {det['confidence']:.3f}\n")
            else:
                f.write("No detections\n")

            f.write("-"*70 + "\n\n")

            # 绘制和保存
            img_vis = img.copy()
            for det in detections:
                cx, cy = det['x'], det['y']
                bw, bh = det['w'], det['h']
                conf = det['confidence']
                class_name = det['class_name']

                if det['class'] == 0:  # Tennis - green circle
                    radius = max(bw, bh) // 2
                    color = (0, 255, 0)
                    cv2.circle(img_vis, (cx, cy), radius, color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(img_vis, label, (cx-40, cy-radius-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:  # Racket - red rectangle
                    x1 = cx - bw // 2
                    y1 = cy - bh // 2
                    x2 = cx + bw // 2
                    y2 = cy + bh // 2
                    color = (0, 0, 255)
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(img_vis, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_path = results_dir / f'{img_path.stem}_pred.jpg'
            cv2.imwrite(str(output_path), img_vis)
            print(f'  ✓ Saved: {output_path.name}\n')

    print("="*70)
    print("Summary")
    print("="*70)
    print(f'Images processed: {len(test_images)}')
    print(f'Total detections: {total_detections}')
    print(f'  - Tennis: {tennis_count}')
    print(f'  - Racket: {racket_count}')
    print(f'\nResults saved to: {results_dir}/')
    print("="*70)

if __name__ == '__main__':
    predict_yolov8()
