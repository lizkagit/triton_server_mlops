import cv2
import numpy as np
from config import COCO_CLASSES

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Ошибка декодирования")
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.transpose(2, 0, 1).astype(np.float32) / 255.0

def postprocess_yolo(output_tensor: np.ndarray, conf_thres=0.25, iou_thres=0.45):
    # Исправление размерности: YOLOv8 часто выдает [1, 84, 8400] вместо [1, 8400, 84]
    if output_tensor.shape[1] < output_tensor.shape[2]:
        output_tensor = np.transpose(output_tensor, (0, 2, 1))
    
    batch_results = []
    for prediction in output_tensor:
        # prediction shape: [8400, 84]
        # Берем только колонки с вероятностями классов (индексы 4-84)
        scores = prediction[:, 4:84]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences >= conf_thres
        if not np.any(mask):
            batch_results.append([])
            continue

        valid_boxes = prediction[mask, :4]
        valid_scores = confidences[mask]
        valid_class_ids = class_ids[mask]

        # Конвертация [cx, cy, w, h] -> [x1, y1, w, h]
        x = valid_boxes[:, 0]
        y = valid_boxes[:, 1]
        w = valid_boxes[:, 2]
        h = valid_boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2

        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, w, h], axis=1).tolist(),
            scores=valid_scores.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append({
                    "class_name": COCO_CLASSES.get(int(valid_class_ids[idx]), "unknown"),
                    "confidence": round(float(valid_scores[idx]), 2),
                    "box": {
                        "x1": int(x1[idx]), "y1": int(y1[idx]),
                        "x2": int(x1[idx] + w[idx]), "y2": int(y1[idx] + h[idx])
                    }
                })
        batch_results.append(detections)
    return batch_results


def postprocess_segmentation(output_tensor, proto_tensor, conf_thres=0.25, iou_thres=0.45):
    # Исправление размерности
    if output_tensor.shape[1] < output_tensor.shape[2]:
        output_tensor = np.transpose(output_tensor, (0, 2, 1))  # [batch, 8400, 116]
    
    batch_results = []
    
    for i in range(output_tensor.shape[0]):
        prediction = output_tensor[i]  # [8400, 116]
        proto = proto_tensor[i]  # [32, 160, 160]
        
        # 1. Извлекаем вероятности классов и уверенности
        scores = prediction[:, 4:84]  # колонки с классами
        confidences = np.max(scores, axis=1)  # максимальная уверенность для каждого предсказания
        class_ids = np.argmax(scores, axis=1)  # ID классов
        
        # 2. Фильтрация по порогу уверенности
        mask = confidences >= conf_thres
        if not np.any(mask):
            batch_results.append([])
            continue
        
        valid_preds = prediction[mask]  # [N, 116]
        valid_conf = confidences[mask]  # [N]
        valid_class_ids = class_ids[mask]  # [N]
        
        # 3. Извлекаем bounding boxes для NMS
        boxes = valid_preds[:, :4]  # [N, 4] в формате [cx, cy, w, h]
        
        # Конвертация в формат [x1, y1, w, h]
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        
        # 4. Применяем NMS для удаления дублирующихся детекций
        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, w, h], axis=1).tolist(),
            scores=valid_conf.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )
        
        seg_data = []
        
        if len(indices) > 0:
            # 5. Обрабатываем только лучшие (не дублирующиеся) предсказания
            for idx in indices.flatten():
                # Коэффициенты маски для этого предсказания
                mask_coeffs = valid_preds[idx, 84:].reshape(1, -1)  # [1, 32]
                
                # 6. Генерация маски: линейная комбинация прототипов
                # [1, 32] @ [32, 160*160] -> [1, 160, 160]
                masks = (mask_coeffs @ proto.reshape(32, -1)).reshape(1, 160, 160)
                
                # 7. Сигмоидная активация для получения вероятностей
                masks = 1 / (1 + np.exp(-masks))
                
                # 8. Бинаризация маски и извлечение контура
                m = (masks[0] > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                polygon = []
                if contours:
                    # Берем контур с наибольшей площадью
                    c = max(contours, key=cv2.contourArea)
                    # Масштабируем контур к размеру исходного изображения (640x640)
                    polygon = (c.reshape(-1, 2) * (640 / 160)).tolist()
                
                # 9. Формируем результат для этого объекта
                seg_data.append({
                    "class_name": COCO_CLASSES.get(int(valid_class_ids[idx]), "unknown"),
                    "confidence": round(float(valid_conf[idx]), 3),
                    "mask_polygon": polygon,
                    "box": {  # Добавляем и bounding box для информации
                        "x1": int(x1[idx]), 
                        "y1": int(y1[idx]),
                        "x2": int(x1[idx] + w[idx]), 
                        "y2": int(y1[idx] + h[idx])
                    }
                })
        
        batch_results.append(seg_data)
    
    return batch_results
