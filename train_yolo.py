from ultralytics import YOLO

def main():
    print("Initializing YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    print("Starting training on IAM YOLO Dataset...")
    results = model.train(
        data="YOLO_Dataset/data.yaml",
        epochs=50, 
        imgsz=320, 
        batch=16, 
        device=0, 
        workers=8, 
        amp=True,
        fliplr=0.0,      # no horizontal flip — text direction matters
        flipud=0.0,      # no vertical flip
        degrees=5.0,     # slight rotation for skew
        hsv_h=0.0, 
        hsv_s=0.0, 
        hsv_v=0.05,
        mosaic=0.0,      # disable mosaic — ruins document structure
        translate=0.1,
        name='iam_word_slicer'
    )
    
    print("Training finished! Weights saved in runs/detect/iam_word_slicer/weights/")

if __name__ == '__main__':
    main()
