import time
import logging
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Импортируем модули
from utils import preprocess_image, postprocess_yolo, postprocess_segmentation
from inference import triton_service
from config import MODEL_NAME, MODEL_ENS

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLO Inference Service")

@app.get("/")
async def root():
    return {"status": "ok", "endpoints": ["/infer", "/infer_batch", "/infer_ensemble"]}

@app.post("/infer")
async def infer_single(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        processed_img = preprocess_image(contents)
        batch_input = processed_img[np.newaxis, ...]
        
        # Используем унифицированный метод сервиса
        raw_results = triton_service.run_inference(MODEL_NAME, batch_input, ["output0"])
        detections = postprocess_yolo(raw_results["output0"])[0]
        
        return {
            "status": "success",
            "detected_objects": len(detections),
            "detections": detections
        }
    except Exception as e:
        logger.error(f"Error in /infer: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/infer_batch")
async def infer_batch(images: List[UploadFile] = File(...)):

    if len(images) < 2:
        raise HTTPException(status_code=400, detail="Batch size must be >= 2")
    
    try:
        # 1. Предобработка всех изображений
        processed = []
        for img in images:
            contents = await img.read()
            processed_img = preprocess_image(contents)
            processed.append(processed_img)
        
        batch_input = np.stack(processed, axis=0)
        
        # Вызов детекции
        raw_results = triton_service.run_inference(MODEL_NAME, batch_input, ["output0"])
        
        batch_detections = postprocess_yolo(raw_results["output0"])
        
        results = []
        for i, (detections, img_file) in enumerate(zip(batch_detections, images)):
            results.append({
                "image_index": i,
                "filename": img_file.filename if img_file.filename else f"image_{i}",
                "detected_objects": len(detections),
                "detections": detections
            })
        
        return {
            "status": "success",
            "batch_size": len(images),
            "processing_mode": "batch_detection",
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in /infer_batch: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/infer_ensemble")
async def infer_ensemble(image: UploadFile = File(...)):
    """
    Вызов ансамблевой модели:
    1. Получаем результаты детекции (yolo_det)
    2. Получаем результаты сегментации (yolo_seg)
    3. Обрабатываем оба выхода независимо
    """
    start_time = time.time()
    try:
 
        contents = await image.read()
        processed_img = preprocess_image(contents)
        batch_input = processed_img[np.newaxis, ...]
        
        # Вызов ансамблевой модели Triton
        # Получаем три выхода от ансамбля
        raw_results = triton_service.run_inference(
            MODEL_ENS, 
            batch_input, 
            ["det_output", "seg_output", "proto_output"]
        )
    
        
        # Обработка детекций от yolo_det
        det_results = postprocess_yolo(raw_results["det_output"])[0]
        
        # Обработка сегментации от yolo_seg
        seg_results = postprocess_segmentation(
            raw_results["seg_output"],
            raw_results["proto_output"],
            conf_thres=0.25
        )[0]
        
        return {
            "status": "success",
            "time_ms": round((time.time() - start_time) * 1000, 2),
            "model_used": "ensemble (detection + segmentation)",
            "results": {
                "detection": {   
                    "objects_count": len(det_results),
                    "detections": det_results
                },
                "segmentation": {   
                    "objects_count": len(seg_results),
                    "masks": seg_results
                }
            }
        }
    except Exception as e:
        logger.error(f"Ensemble error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})