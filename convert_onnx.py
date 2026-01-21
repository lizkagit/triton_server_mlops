import os
import onnx
from ultralytics import YOLO

def fix_model_for_triton(input_path, output_path, model_type="det"):

    model = onnx.load(input_path)
    
    # Откатываем версии
    model.ir_version = 8
    while len(model.opset_import) > 0:
        model.opset_import.pop()
    opset = model.opset_import.add()
    opset.domain = "" 
    opset.version = 19

    # Фикс ВХОДА 
    for inp in model.graph.input:
        if "images" in inp.name:
            inp.type.tensor_type.shape.dim.clear()
            # Прописываем [batch, 3, 640, 640]
            inp.type.tensor_type.shape.dim.add().dim_param = "batch"
            inp.type.tensor_type.shape.dim.add().dim_value = 3
            inp.type.tensor_type.shape.dim.add().dim_value = 640
            inp.type.tensor_type.shape.dim.add().dim_value = 640
            print(f" Вход {inp.name} зафиксирован как [-1, 3, 640, 640]")

    #  Фикс ВЫХОДОВ (output0, output1)
    for out in model.graph.output:
        if "output0" in out.name:
            out.type.tensor_type.shape.dim.clear()
            out.type.tensor_type.shape.dim.add().dim_param = "batch"
            # Для детекции/сегментации yolo11 на 640px это обычно 84 или 116 строк и 8400 колонок
            out.type.tensor_type.shape.dim.add().dim_value = 84 if model_type == "det" else 116
            out.type.tensor_type.shape.dim.add().dim_value = 8400
            print(f"Выход {out.name} зафиксирован")
            
        if "output1" in out.name: # Только для сегментации
            out.type.tensor_type.shape.dim.clear()
            out.type.tensor_type.shape.dim.add().dim_param = "batch"
            out.type.tensor_type.shape.dim.add().dim_value = 32
            out.type.tensor_type.shape.dim.add().dim_value = 160
            out.type.tensor_type.shape.dim.add().dim_value = 160
            print(f"Выход {out.name} (mask prototypes) зафиксирован")

    onnx.save(model, output_path)

def download_and_prepare(model_variant="yolo11n.pt", task="det"):
    print(f"\n🚀 Работаем с {model_variant} (задача: {task})")
    
    # Отключаем simplify, чтобы он не задирал Opset до 22
    model = YOLO(model_variant)
    model.export(
        format="onnx", 
        imgsz=640, 
        opset=19, 
        simplify=False, 
        dynamic=True     # Просим Ultralytics саму сделать динамику 
    )
    
    generated_onnx = model_variant.replace(".pt", ".onnx")
    final_destination = f"triton_model_repo/yolo_{task}/1/model.onnx"
    
 
    fix_model_for_triton(generated_onnx, final_destination, model_type=task)
    
    # Удаляем временный файл
    if os.path.exists(generated_onnx):
        os.remove(generated_onnx)

if __name__ == "__main__":
    # Детекция
    download_and_prepare("yolo11n.pt", "det")
    # Сегментация
    download_and_prepare("yolo11n-seg.pt", "seg")